"""Common transverse-coupling sweep for an aligned three-nitrogen toy RPM.

N5 and N10 couple to electron A; N1 couples to electron B.  All nuclei are
spin 1.  The original two-nucleus control retains transverse N5 and axial N1.
The full pair is diagonalized (108 states for three nuclei, 36 for control),
and equal-rate yields are integrated analytically in the eigenbasis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate
import scipy.linalg
import scipy.optimize

from rpm_quantum_needle_model import GAMMA_E_MHZ_PER_MT, TWOPI, add_quadrupole_argument, spin_operators
from rpm_statevector_engine import dense_aligned_toy_quadrupole

LIFETIMES_US = np.array([1.0, 10.0, 100.0])
FIELD_MT = 0.05
# Retain the original toy's N5 and N1 axial values.  N10/N1 transverse
# principal values are axial averages from the brief's a_iso,T table.
NITROGENS = (
    ("FAD:N5", 0, -0.0989, 1.7569),
    ("FAD:N10", 0, -0.0193, 0.6046),
    ("Trp:N1", 1, -0.0583, 1.0812),
)
B_N5_AT_ONE_UT = 1000.0 * math.sqrt(2.0) * abs(NITROGENS[0][2])
MODELS = ("two_nucleus_control", "three_nitrogen")


def embed(op: np.ndarray, axis: int, dims: tuple[int, ...]) -> np.ndarray:
    value = np.ones((1, 1), dtype=complex)
    for i, d in enumerate(dims):
        value = np.kron(value, op if i == axis else np.eye(d))
    return value


class PairModel:
    def __init__(self, name: str, field_mT: float = FIELD_MT,
                 n5_azz_mT: float | None = None, axial_scale: float = 1.0,
                 quadrupole_scale: float = 0.0):
        if name not in MODELS:
            raise ValueError(name)
        self.name = name
        self.nuclei = NITROGENS if name == "three_nitrogen" else (NITROGENS[0], NITROGENS[2])
        if n5_azz_mT is not None:
            self.nuclei = tuple((label,electron,aperp,float(n5_azz_mT) if label=="FAD:N5" else azz) for label,electron,aperp,azz in self.nuclei)
        if not math.isfinite(axial_scale) or axial_scale <= 0:
            raise ValueError("axial_scale must be finite and positive")
        self.nuclei = tuple((label, electron, aperp, azz * axial_scale)
                            for label, electron, aperp, azz in self.nuclei)
        self.dims = (2, 2) + (3,) * len(self.nuclei)
        self.nuclear_dimension = 3 ** len(self.nuclei)
        self.dimension = 4 * self.nuclear_dimension
        self.quadrupole = (
            quadrupole_scale * dense_aligned_toy_quadrupole(
                self.dims, [label.replace("Trp:", "TrpH:") for label, *_ in self.nuclei]
            ) if quadrupole_scale else None
        )
        electrons = [tuple(embed(op, e, self.dims) for op in spin_operators(0.5)) for e in (0, 1)]
        self.zeeman_x = (TWOPI * GAMMA_E_MHZ_PER_MT * field_mT * (electrons[0][0] + electrons[1][0])).real
        self.zeeman_z = (TWOPI * GAMMA_E_MHZ_PER_MT * field_mT * (electrons[0][2] + electrons[1][2])).real
        self.axial = np.zeros((self.dimension, self.dimension))
        self.transverse = self.axial.copy()
        for j, (label, electron, aperp, azz) in enumerate(self.nuclei):
            nuclear = tuple(embed(op, j + 2, self.dims) for op in spin_operators(1.0))
            self.axial += (TWOPI * GAMMA_E_MHZ_PER_MT * azz * (electrons[electron][2] @ nuclear[2])).real
            if name == "two_nucleus_control" and label == "Trp:N1":
                continue
            self.transverse += (TWOPI * GAMMA_E_MHZ_PER_MT * aperp * (
                electrons[electron][0] @ nuclear[0] + electrons[electron][1] @ nuclear[1]
            )).real

    def hamiltonian(self, scale: float, theta_deg: float) -> np.ndarray:
        theta = math.radians(theta_deg)
        h = self.axial + scale * self.transverse + math.sin(theta) * self.zeeman_x + math.cos(theta) * self.zeeman_z
        return h if self.quadrupole is None else h + self.quadrupole

    def yields_from_hamiltonian(self, h: np.ndarray, lifetimes: np.ndarray) -> np.ndarray:
        energies, vectors = scipy.linalg.eigh(h, overwrite_a=True, check_finite=False, driver="evd")
        d = self.nuclear_dimension
        singlet_rows = (vectors[d:2*d] - vectors[2*d:3*d]) / math.sqrt(2.0)
        # Real-valued aligned tensors are a special case. Rotated nuclear
        # quadrupoles can give complex eigenvectors, requiring a dagger and
        # absolute squares rather than a transpose and ordinary squares.
        projector_eigenbasis = singlet_rows.conj().T @ singlet_rows
        weights = np.abs(projector_eigenbasis) ** 2 / d
        gaps_squared = (energies[:, None] - energies[None, :]) ** 2
        return np.array([np.sum(weights / (1.0 + float(tau)**2 * gaps_squared)) for tau in lifetimes])

    def curve(self, scale: float, theta: np.ndarray, lifetimes: np.ndarray | None = None) -> np.ndarray:
        lifetimes = LIFETIMES_US if lifetimes is None else lifetimes
        return np.array([self.yields_from_hamiltonian(self.hamiltonian(scale, float(t)), lifetimes) for t in theta]).T


def scale_grid() -> np.ndarray:
    # A common, well-resolved grid for both models; include exact zero and
    # unscaled tensors.  This extends through the turnover range in Figure 2.
    return np.unique(np.concatenate(([0.0, 1.0], np.geomspace(0.175, 4.0, 41))))


def metrics_for_scale(model_name: str, scale: float, angle_step: float, refine_step: float = 0.005,
                      quadrupole_scale: float = 0.0) -> dict:
    start = time.perf_counter()
    model = PairModel(model_name, quadrupole_scale=quadrupole_scale)
    theta = np.arange(0.0, 90.0 + angle_step / 2.0, angle_step)
    curves = model.curve(scale, theta)
    metrics = []
    for i, tau in enumerate(LIFETIMES_US):
        slope = np.abs(np.gradient(curves[i], np.deg2rad(theta), edge_order=2))
        # Refine several separated angular maxima so competing features can
        # win; the scan does not preselect a needle branch.
        candidates = []
        for index in np.argsort(slope)[::-1]:
            t = float(theta[index])
            if all(abs(t - c) >= 0.2 for c in candidates):
                candidates.append(t)
            if len(candidates) == 3:
                break
        best_gamma, best_theta = float(slope.max()), float(theta[slope.argmax()])
        for center in candidates:
            lower, upper = max(0.0, center - 0.15), min(90.0, center + 0.15)
            local_theta = np.arange(lower, upper + 0.5 * refine_step, refine_step)
            local_y = model.curve(scale, local_theta)[i]
            spline = scipy.interpolate.CubicSpline(np.deg2rad(local_theta), local_y)
            probes = np.deg2rad(np.arange(lower, upper + 0.00025, 0.0005))
            index = int(np.argmax(np.abs(spline(probes, 1))))
            lo, hi = probes[max(0, index-1)], probes[min(len(probes)-1, index+1)]
            optimum = scipy.optimize.minimize_scalar(lambda t: -abs(float(spline(t, 1))), bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
            gamma = float(-optimum.fun)
            if gamma > best_gamma:
                best_gamma, best_theta = gamma, float(np.rad2deg(optimum.x))
        shoulders = model.curve(scale, np.array([0.0, 80.0, 85.0, 90.0]))[i]
        metrics.append({
            "model": model_name, "lambda": float(scale), "tau_us": float(tau),
            "b_N5_uT": scale * B_N5_AT_ONE_UT,
            "b_N10_uT": scale * 1000 * math.sqrt(2) * abs(NITROGENS[1][2]) if model_name == "three_nitrogen" else 0.0,
            "b_N1_uT": scale * 1000 * math.sqrt(2) * abs(NITROGENS[2][2]) if model_name == "three_nitrogen" else 0.0,
            "gamma_per_rad": best_gamma, "theta_gamma_deg": best_theta,
            "needle_80_to_90": float(shoulders[1] - shoulders[3]),
            "needle_85_to_90": float(shoulders[2] - shoulders[3]),
            "contrast_0_to_90": float(shoulders[0] - shoulders[3]),
            "angular_range": float(np.ptp(curves[i])),
            "phi_90": float(shoulders[3]), "J_uT": 0.0, "D_uT": 0.0, "quadrupole_scale": quadrupole_scale,
        })
    return {"model": model_name, "scale": float(scale), "theta_deg": theta, "curves": curves, "metrics": metrics, "elapsed_seconds": time.perf_counter()-start}


def checkpoint_path(output: Path, model: str, scale: float) -> Path:
    return output / "checkpoints" / f"{model}_lambda{scale:.10f}.npz"


def save_cell(output: Path, cell: dict) -> None:
    path = checkpoint_path(output, cell["model"], cell["scale"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, theta_deg=cell["theta_deg"], curves=cell["curves"], metrics_json=json.dumps(cell["metrics"]), elapsed_seconds=cell["elapsed_seconds"])
    temp.replace(path)


def load_rows(output: Path) -> list[dict]:
    rows = []
    for path in sorted((output / "checkpoints").glob("*.npz")):
        with np.load(path) as data:
            rows.extend(json.loads(data["metrics_json"].item()))
    return sorted(rows, key=lambda r: (r["model"], r["tau_us"], r["lambda"]))


def refine_peak(model: str, tau: float, metric: str, lower: float, upper: float, angle_step: float,
                quadrupole_scale: float = 0.0) -> dict:
    cells = {}
    def objective(scale):
        cell=metrics_for_scale(model,float(scale),angle_step,quadrupole_scale=quadrupole_scale)
        cells[float(scale)]=cell
        row=next(r for r in cell["metrics"] if r["tau_us"]==tau)
        return -row[metric]
    optimum=scipy.optimize.minimize_scalar(objective,bounds=(lower,upper),method="bounded",options={"xatol":0.001,"maxiter":20})
    if not optimum.success:
        raise RuntimeError(f"peak refinement failed: {model}, tau={tau}, {metric}")
    return {"model":model,"tau":tau,"metric":metric,"lambda":float(optimum.x),"cells":list(cells.values())}


def refine_sweep(output: Path, angle_step: float, workers: int, quadrupole_scale: float = 0.0) -> None:
    marker=output / "peak_refinement.json"
    if marker.exists():
        return
    rows=load_rows(output)
    jobs=[]
    for model in MODELS:
        for tau in (10.,100.):
            selected=sorted([r for r in rows if r["model"]==model and r["tau_us"]==tau],key=lambda r:r["lambda"])
            for metric in ("gamma_per_rad","needle_80_to_90"):
                index=max(range(len(selected)),key=lambda i:selected[i][metric])
                if index in (0,len(selected)-1):
                    continue
                jobs.append((model,tau,metric,selected[index-1]["lambda"],selected[index+1]["lambda"],angle_step,quadrupole_scale))
    diagnostics=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(refine_peak,*job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            for cell in result.pop("cells"):
                save_cell(output,cell)
            diagnostics.append(result)
            print(f"refined [{i}/{len(jobs)}] {result['model']} tau={result['tau']:g} {result['metric']}: lambda={result['lambda']:.5f}",flush=True)
    marker.write_text(json.dumps(diagnostics,indent=2)+"\n")


def report(output: Path, quadrupole_scale: float = 0.0) -> None:
    rows = load_rows(output)
    with (output / "sweep_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    peaks = []
    for model in MODELS:
        for tau in LIFETIMES_US:
            selected = [r for r in rows if r["model"] == model and r["tau_us"] == tau]
            sensitivity_peak = max(selected, key=lambda r:r["gamma_per_rad"])
            amplitude_peak = max(selected, key=lambda r:r["needle_80_to_90"])
            baseline = min(selected, key=lambda r:abs(r["lambda"]-1.0))
            peaks.append({"model":model,"tau_us":float(tau),"sensitivity_peak":sensitivity_peak,"amplitude_peak":amplitude_peak,"unscaled_tensors":baseline,"interior_sensitivity_peak":0 < sensitivity_peak["lambda"] < 4})
    summary = {"peaks":peaks,"precision":"Exact nuclear trace and eigenbasis yield; angular derivatives use local 0.005-degree refinement. Coupling maxima located on 41 logarithmic scales and refined to lambda tolerance 0.001 at tau=10,100 us.","amplitude_definition":"Phi_S(80 deg)-Phi_S(90 deg); signed. The brief does not explicitly specify its amplitude estimator; this definition is reported rather than assumed."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.15})
    colors = {1.0:"#7f7f7f",10.0:"#e67e22",100.0:"#287bb5"}
    fig, axes = plt.subplots(1,2,figsize=(11.4,4.5),constrained_layout=True)
    for tau in LIFETIMES_US:
        for model in MODELS:
            selected = [r for r in rows if r["model"]==model and r["tau_us"]==tau and r["lambda"]>0]
            x = [r["b_N5_uT"] for r in selected]
            style = "--" if model=="two_nucleus_control" else "-"
            label = rf"$\tau={tau:g}\,\mu$s" if model=="three_nitrogen" else None
            axes[0].plot(x,[r["needle_80_to_90"] for r in selected],style,color=colors[tau],label=label,lw=1.5)
            axes[1].plot(x,[r["gamma_per_rad"] for r in selected],style,color=colors[tau],label=label,lw=1.5)
        peak = next(p for p in peaks if p["model"]=="three_nitrogen" and p["tau_us"]==tau)
        for ax, key, ykey in ((axes[0],"amplitude_peak","needle_80_to_90"),(axes[1],"sensitivity_peak","gamma_per_rad")):
            r = peak[key]
            if 0 < r["lambda"] < 4:
                ax.scatter(r["b_N5_uT"],r[ykey],marker="*",s=160,color=colors[tau],edgecolor="black",linewidth=0.5,zorder=5)
    axes[0].set(xscale="log",xlabel=r"N5-equivalent transverse coupling $b_{N5}$ ($\mu$T)",ylabel=r"needle contrast $\Phi_S(80^\circ)-\Phi_S(90^\circ)$",title="(a) Needle amplitude")
    axes[1].set(xscale="log",yscale="log",xlabel=r"N5-equivalent transverse coupling $b_{N5}$ ($\mu$T)",ylabel=r"$\Gamma=\max_\theta|\partial\Phi_S/\partial\theta|$ (rad$^{-1}$)",title="(b) Angular sensitivity")
    for ax in axes:
        ax.axvline(B_N5_AT_ONE_UT,color="#b83243",ls=":",lw=1.3)
        ax.legend(fontsize=9)
    fig.suptitle("Transverse-coupling optimum with three anisotropic nitrogens")
    fig.supxlabel(r"solid: N5 + N10 on FAD, N1 on Trp; dashed: two-nucleus control; $B_0=50\,\mu$T, $J=D=0$"
                  + f", Q scale={quadrupole_scale:g}",fontsize=9)
    for suffix in ("png","pdf"):
        fig.savefig(output / f"three_nitrogen_optimum.{suffix}",dpi=300,facecolor="white")
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12.7,4.5),constrained_layout=True)
    for ax,tau in zip(axes,LIFETIMES_US):
        p=next(p for p in peaks if p["model"]=="three_nitrogen" and p["tau_us"]==tau)
        conditions=[(0.,"zero transverse","#7f7f7f"),(1.,"unscaled tensors","#e67e22")]
        peak_scale=p["sensitivity_peak"]["lambda"]
        if peak_scale == 0.0:
            conditions[0]=(0.,"zero transverse (maximum sensitivity)","#287bb5")
        else:
            conditions.append((peak_scale,"maximum sensitivity","#287bb5"))
        for scale,label,color in conditions:
            path=checkpoint_path(output,"three_nitrogen",scale)
            with np.load(path) as data:
                t=data["theta_deg"]; y=data["curves"][int(np.flatnonzero(LIFETIMES_US==tau)[0])]
            ax.plot(np.concatenate((t,180-t[-2::-1])),np.concatenate((y,y[-2::-1])),color=color,label=label)
        ax.set(xlabel=r"field angle $\theta$ (degrees)",ylabel=r"singlet yield $\Phi_S$",title=rf"$\tau={tau:g}\,\mu$s")
        ax.legend(fontsize=8)
    fig.suptitle(f"Three-nitrogen angular response across coherence times (Q scale={quadrupole_scale:g})")
    for suffix in ("png","pdf"):
        fig.savefig(output / f"three_nitrogen_angular_curves.{suffix}",dpi=300,facecolor="white")
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/three_nitrogen_optimum"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--angle-step",type=float,default=0.05)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    add_quadrupole_argument(parser, default=0.0)
    args=parser.parse_args()
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    config={"nuclei":NITROGENS,"axis_convention":"all principal z axes aligned; diagonal axial tensors","dimensions":[2,2,3,3,3],"field_mT":FIELD_MT,"lifetimes_us":LIFETIMES_US.tolist(),"scale_grid":scale_grid().tolist(),"angle_step_deg":args.angle_step,"refine_step_deg":0.005,"J_uT":0.,"D_uT":0.,"quadrupole_scale":0.,"initial_state":"electron singlet and exact maximally mixed nuclear state","method":"full pair dense eigendecomposition, analytic equal-rate recombination integral"}
    config["quadrupole_scale"] = args.quadrupole_scale
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest=output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["configuration_hash"]!=config_hash:
        raise ValueError("incompatible output manifest")
    if not manifest.exists():
        manifest.write_text(json.dumps({"configuration":config,"configuration_hash":config_hash,"environment":{"host":platform.node(),"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}},indent=2)+"\n")
    if not args.report_only:
        pending=[(m,float(s)) for m in MODELS for s in scale_grid() if not (args.resume and checkpoint_path(output,m,float(s)).exists())]
        print(f"starting {len(pending)} scale cells, {args.workers} workers; each includes all three lifetimes",flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(metrics_for_scale,m,s,args.angle_step,quadrupole_scale=args.quadrupole_scale):(m,s) for m,s in pending}
            for i,future in enumerate(as_completed(futures),1):
                cell=future.result()
                save_cell(output,cell)
                print(f"[{i}/{len(pending)}] {cell['model']} lambda={cell['scale']:.5f}, {cell['elapsed_seconds']:.1f}s",flush=True)
    missing=[(m,float(s)) for m in MODELS for s in scale_grid() if not checkpoint_path(output,m,float(s)).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} missing scale cells")
    if not args.report_only:
        refine_sweep(output,args.angle_step,args.workers,args.quadrupole_scale)
    report(output,args.quadrupole_scale)
    print(f"complete: {output}",flush=True)


if __name__=="__main__":
    main()
