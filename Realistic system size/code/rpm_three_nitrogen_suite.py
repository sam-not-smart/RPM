"""Remaining transverse/axial brief experiments in the aligned three-N toy.

All three axial components scale together; all transverse components scale
together independently. No radical factorization, sampling or time stepping.
The completed first transverse-coupling sweep is retained separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from rpm_three_nitrogen_optimum import PairModel, NITROGENS, B_N5_AT_ONE_UT
from rpm_quantum_needle_model import add_quadrupole_argument

AXIAL_N5_MT = np.array([0.6046, 1.2, 1.7569])
TAUS = np.array([5., 10., 20., 50., 100.])
DIRECT_TAU = 29.
DIRECT_B = np.array([70., 90., 110., 125., 132.5, 140., 160., 185., 215.])
REFERENCE_C = 391.


def field_grid():
    return np.unique(np.round(np.r_[np.geomspace(.05, 100., 41),
        np.arange(.4, .80001, .025), [.05, .2, .5, .6, .7, 1., 10., 100.]], 10))


def transverse_grid():
    return np.unique(np.r_[0., 1., np.geomspace(.04, 4., 41)])


def axial_crossing_seeds(model, field):
    """Only angular search seeds. Every evaluated H is the full pair matrix."""
    values = []
    for m in itertools.product((-1, 0, 1), repeat=3):
        total = sum(mi * nucleus[3] for mi, nucleus in zip(m, model.nuclei))
        cosine = -total / (2 * field)
        if 0 <= cosine <= 1:
            values.append(math.degrees(math.acos(cosine)))
    return np.unique(np.round(values, 9)).tolist()


def slope_max(theta, phi):
    spline = CubicSpline(np.deg2rad(theta), phi)
    stationary = spline.derivative(2).roots(extrapolate=False)
    probes = np.r_[np.deg2rad(theta), stationary]
    gradients = np.abs(spline(probes, 1))
    index = int(np.argmax(gradients))
    return float(gradients[index]), float(np.rad2deg(probes[index]))


def angular_metrics(model, scale, taus, angle_step=.05, refine_step=.002,
                    crossing_seeds=(), half_width=.10):
    theta = np.linspace(0, 90, int(round(90 / angle_step)) + 1)
    curves = model.curve(scale, theta, np.asarray(taus))
    candidates = list(crossing_seeds)
    gamma = np.empty(len(taus))
    angles = np.empty(len(taus))
    for i, curve in enumerate(curves):
        derivative = np.abs(np.gradient(curve, np.deg2rad(theta), edge_order=2))
        selected = []
        for index in np.argsort(derivative)[::-1]:
            t = float(theta[index])
            if all(abs(t - c) > 2 * half_width for c in selected):
                selected.append(t)
            if len(selected) == 4:
                break
        candidates.extend(selected)
        # Coarse interpolation is only a search aid: an unresolved resonance
        # can overshoot, so never retain its derivative as the final maximum.
        gamma[i], angles[i] = 0., math.nan
    # Combine overlapping neighborhoods once; reuse the same eigensystems
    # for every lifetime rather than recalculating a curve per tau.
    intervals = []
    for center in sorted(candidates):
        lower, upper = max(0., center - half_width), min(90., center + half_width)
        if intervals and lower <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], upper)
        else:
            intervals.append([lower, upper])
    for lower, upper in intervals:
        local_t = np.linspace(lower, upper, max(4, int(math.ceil((upper-lower)/refine_step))+1))
        local = model.curve(scale, local_t, np.asarray(taus))
        for i, phi in enumerate(local):
            g, a = slope_max(local_t, phi)
            if g > gamma[i]:
                gamma[i], angles[i] = g, a
    return {"theta_deg":theta, "curves":curves,
            "gamma":gamma, "theta_gamma_deg":angles}


def field_cell(field, angle_step=.01, refine_factor=1., quadrupole_scale=0.):
    start = time.perf_counter()
    model = PairModel("three_nitrogen", field_mT=field, quadrupole_scale=quadrupole_scale)
    fine = (.0002 if field >= 3 else .001) * refine_factor
    seeds = axial_crossing_seeds(model, field) if field >= .3 else []
    results = [angular_metrics(model, scale, [10.], angle_step, fine,
                              seeds, .025) for scale in (1., 0.)]
    row = {"field_mT":float(field), "tau_us":10.,
           "gamma_full_per_rad":float(results[0]["gamma"][0]),
           "gamma_ablated_per_rad":float(results[1]["gamma"][0]),
           "theta_full_deg":float(results[0]["theta_gamma_deg"][0]),
           "theta_ablated_deg":float(results[1]["theta_gamma_deg"][0]),
           "ablation_fraction":float(1-results[1]["gamma"][0]/results[0]["gamma"][0]),
           "local_angle_step_deg":fine}
    return {"key":f"field_B{field:.10f}", "rows":[row],
            "theta_deg":results[0]["theta_deg"],
            "curves":np.array([r["curves"][0] for r in results]),
            "elapsed_seconds":time.perf_counter()-start}


def scaling_cell(azz, scale, angle_step=.05, refine_step=.002, taus=None, kind="scaling",
                 quadrupole_scale=0.):
    start = time.perf_counter()
    taus = TAUS if taus is None else np.asarray(taus)
    axial_scale = azz / NITROGENS[0][3]
    model = PairModel("three_nitrogen", axial_scale=axial_scale, quadrupole_scale=quadrupole_scale)
    result = angular_metrics(model, scale, taus, angle_step, refine_step)
    rows = [{"Azz_N5_mT":float(azz), "axial_scale":float(axial_scale),
             "Azz_N10_mT":float(model.nuclei[1][3]),
             "Azz_N1_mT":float(model.nuclei[2][3]),
             "lambda":float(scale), "b_N5_uT":float(scale*B_N5_AT_ONE_UT),
             "tau_us":float(tau), "gamma_per_rad":float(result["gamma"][i]),
             "theta_gamma_deg":float(result["theta_gamma_deg"][i]),
             "angular_range":float(np.ptp(result["curves"][i]))}
            for i, tau in enumerate(taus)]
    return {"key":f"{kind}_A{azz:.8f}_lambda{scale:.10f}", "rows":rows,
            "theta_deg":result["theta_deg"], "curves":result["curves"],
            "elapsed_seconds":time.perf_counter()-start}


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def save_cell(output, cell):
    path = output / "checkpoints" / (cell["key"]+".npz")
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, theta_deg=cell["theta_deg"], curves=cell["curves"],
        rows_json=json.dumps(cell["rows"]), elapsed_seconds=cell["elapsed_seconds"])
    temporary.replace(path)


def load_rows(output, prefix):
    rows = []
    for path in sorted((output/"checkpoints").glob(prefix+"*.npz")):
        if ".tmp." not in path.name:
            with np.load(path) as data:
                rows.extend(json.loads(data["rows_json"].item()))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_jobs(output, jobs, workers, resume):
    pending = [job for job in jobs if not (resume and (output/"checkpoints"/(job[0]+".npz")).exists())]
    print(f"{len(pending)} pending units, {workers} single-threaded workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args):key for key, fn, args in pending}
        for i, future in enumerate(as_completed(futures), 1):
            cell = future.result()
            if cell["key"] != futures[future]:
                raise ValueError("checkpoint key mismatch")
            save_cell(output, cell)
            print(f"[{i}/{len(pending)}] {cell['key']} ({cell['elapsed_seconds']:.1f}s)", flush=True)


def local_peak_brackets(rows):
    """Refine all substantial local maxima, not a preselected needle branch."""
    ordered = sorted(rows, key=lambda r:r["lambda"])
    g = np.array([r["gamma_per_rad"] for r in ordered])
    return [(ordered[i-1]["lambda"], ordered[i+1]["lambda"])
            for i in range(1, len(ordered)-1)
            if g[i] >= g[i-1] and g[i] >= g[i+1] and g[i] >= .9*g.max()]


def refine_optimum(azz, tau, bracket, kind="scaling", quadrupole_scale=0.):
    cells = {}
    def objective(scale):
        cell = scaling_cell(azz, float(scale), taus=[tau], kind=kind, quadrupole_scale=quadrupole_scale)
        cells[float(scale)] = cell
        return -cell["rows"][0]["gamma_per_rad"]
    optimum = minimize_scalar(objective, bounds=bracket, method="bounded",
                              options={"xatol":.0005, "maxiter":30})
    if not optimum.success:
        raise RuntimeError("coupling maximum refinement failed")
    return {"Azz_N5_mT":azz, "tau_us":tau, "bracket":list(bracket),
            "lambda":float(optimum.x), "cells":list(cells.values())}


def refine_all(output, workers, kind="scaling", quadrupole_scale=0.):
    marker = output / f"{kind}_refinement.json"
    if marker.exists():
        return
    rows = load_rows(output, kind+"_")
    tasks = []
    for azz, tau in sorted({(r["Azz_N5_mT"], r["tau_us"]) for r in rows}):
        selected = [r for r in rows if r["Azz_N5_mT"]==azz and r["tau_us"]==tau]
        tasks.extend((azz, tau, bracket, kind, quadrupole_scale) for bracket in local_peak_brackets(selected))
    diagnostics = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(refine_optimum, *args) for args in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            for cell in result.pop("cells"):
                # Separate refinements from discovery keys: a refined cell
                # contains one tau, discovery cells contain all five.
                cell["key"] += f"_tau{result['tau_us']:g}"
                save_cell(output, cell)
            diagnostics.append(result)
            print(f"refined [{i}/{len(tasks)}] Azz={result['Azz_N5_mT']:g} tau={result['tau_us']:g} lambda={result['lambda']:.5f}",flush=True)
    atomic_json(marker, diagnostics)


def best_scaling(rows):
    return [max((r for r in rows if r["Azz_N5_mT"]==azz and r["tau_us"]==tau),
                key=lambda r:r["gamma_per_rad"])
            for azz in AXIAL_N5_MT for tau in TAUS]


def resolution_checks(output, workers, quadrupole_scale=0.):
    marker = output / "resolution_checks.json"
    if marker.exists():
        return json.loads(marker.read_text())
    field_rows = load_rows(output, "field_")
    peaks = best_scaling(load_rows(output, "scaling_"))
    diagnostics = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for field in (.05, .6, 100.):
            original = next(r for r in field_rows if abs(r["field_mT"]-field)<1e-8)
            futures[pool.submit(field_cell, field, .005, .5, quadrupole_scale)] = ("field", original)
        for original in peaks:
            futures[pool.submit(scaling_cell, original["Azz_N5_mT"], original["lambda"], .025, .001, [original["tau_us"]], "verify", quadrupole_scale)] = ("scaling", original)
        for future in as_completed(futures):
            category, original = futures[future]
            cell = future.result()
            refined = cell["rows"][0]
            keys = ["gamma_full_per_rad", "gamma_ablated_per_rad"] if category=="field" else ["gamma_per_rad"]
            changes = {k:abs(refined[k]/original[k]-1) for k in keys}
            diagnostics.append({"category":category, "original":original,
                                "refined":refined, "relative_gamma_change":changes,
                                "passed_0p5_percent":max(changes.values()) <= .005})
            print(f"resolution {category}: maximum relative change {max(changes.values()):.3g}",flush=True)
    atomic_json(marker, diagnostics)
    return diagnostics


def scaling_fit(peaks):
    fits = []
    for azz in AXIAL_N5_MT:
        selected = [r for r in peaks if r["Azz_N5_mT"]==azz]
        t = np.array([r["tau_us"] for r in selected])
        b = np.array([r["b_N5_uT"] for r in selected])
        if np.any(b <= 0):
            raise ValueError("no positive optimum; power law is not defined")
        exponent, intercept = np.polyfit(np.log(t), np.log(b/azz), 1)
        fitted = np.exp(intercept) * t**exponent
        observed_log = np.log(b/azz)
        fitted_log = np.log(fitted)
        fits.append({"Azz_N5_mT":float(azz), "exponent":float(exponent),
                     "coefficient":float(np.exp(intercept)),
                     "log_R_squared":float(1-np.sum((observed_log-fitted_log)**2)/np.sum((observed_log-observed_log.mean())**2)),
                     "max_relative_power_fit_residual":float(np.max(abs((b/azz)/fitted-1)))})
    effective_c = np.array([r["b_N5_uT"]*math.sqrt(r["tau_us"])/r["Azz_N5_mT"] for r in peaks])
    c = float(np.exp(np.mean(np.log(effective_c))))
    residual = effective_c/c-1
    spread = []
    for tau in TAUS:
        normalized = np.array([r["b_N5_uT"]/r["Azz_N5_mT"] for r in peaks if r["tau_us"]==tau])
        spread.append({"tau_us":float(tau), "relative_range":float(np.ptp(normalized)/normalized.mean())})
    return {"per_axial_scale_power_fits":fits,"fixed_minus_half_C":c,
            "max_relative_fixed_law_residual":float(np.max(abs(residual))),
            "normalized_curve_spread":spread,
            "passes_5_percent_collapse":max(r["relative_range"] for r in spread)<=.05,
            "passes_5_percent_fixed_minus_half_law":float(np.max(abs(residual)))<=.05,
            "C_eff_min":float(effective_c.min()),"C_eff_max":float(effective_c.max())}


def save_figure(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def shade_regimes(axes, fields):
    """Shade the measured diagnostic, allowing more than one crossover."""
    boundaries = [fields[0]["field_mT"]]
    boundaries += [math.sqrt(a["field_mT"]*b["field_mT"])
                   for a,b in zip(fields[:-1],fields[1:])]
    boundaries += [fields[-1]["field_mT"]]
    segments = []
    for i,row in enumerate(fields):
        transverse = row["ablation_fraction"]>.5
        if segments and segments[-1][2]==transverse:
            segments[-1][1]=boundaries[i+1]
        else:
            segments.append([boundaries[i],boundaries[i+1],transverse])
    for ax in axes:
        for lower,upper,transverse in segments:
            ax.axvspan(lower,upper,color="#bedcef" if transverse else "#f0c5d8",alpha=.25,zorder=0)
        for transverse,label in ((True,"transverse dominated"),(False,"axial response survives")):
            choices=[s for s in segments if s[2]==transverse]
            if choices:
                lower,upper,_=max(choices,key=lambda s:math.log(s[1]/s[0]))
                ax.text(math.sqrt(lower*upper),.025,label,
                        transform=ax.get_xaxis_transform(),ha="center",va="bottom",
                        fontsize=8,color="#386584" if transverse else "#8b4565")


def require_complete_inventory(output):
    expected = [f"field_B{b:.10f}" for b in field_grid()]
    expected += [f"scaling_A{a:.8f}_lambda{s:.10f}"
                 for a in AXIAL_N5_MT for s in transverse_grid()]
    expected += [f"direct_A1.75690000_lambda{s:.10f}"
                 for s in np.unique(np.r_[transverse_grid(), DIRECT_B/B_N5_AT_ONE_UT])]
    missing = [key for key in expected if not (output/"checkpoints"/(key+".npz")).exists()]
    if missing:
        raise RuntimeError(f"incomplete suite: {len(missing)} missing discovery units")


def report(output, quadrupole_scale=0.):
    require_complete_inventory(output)
    repo_root = Path(__file__).resolve().parents[2]
    brief = repo_root/"methods"/"references"/"RPM_transverse_vs_axial_brief_v2.pdf"
    if not brief.is_file():
        brief = repo_root/"Paper"/"Final Results"/"RPM_transverse_vs_axial_brief_v2.pdf"
    atomic_json(output/"analysis_metadata.json",{
        "reported_at_UTC":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "engine_source_hash":hashlib.sha256(Path(__file__).with_name("rpm_three_nitrogen_optimum.py").read_bytes()).hexdigest(),
        "runner_source_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_brief_hash":hashlib.sha256(brief.read_bytes()).hexdigest(),
        "host":platform.node(),"python":platform.python_version(),
        "numpy":np.__version__,"scipy":scipy.__version__})
    fields = sorted(load_rows(output,"field_"),key=lambda r:r["field_mT"])
    scaling = load_rows(output,"scaling_")
    direct = sorted(load_rows(output,"direct_"),key=lambda r:r["lambda"])
    if not fields or not scaling or not direct:
        raise RuntimeError("complete all three stages before reporting")
    peaks = best_scaling(scaling)
    fit = scaling_fit(peaks)
    direct_peak = max(direct, key=lambda r:r["gamma_per_rad"])
    direct_real = min(direct, key=lambda r:abs(r["b_N5_uT"]-132.5))
    checks = json.loads((output/"resolution_checks.json").read_text())
    crossings = []
    for a,b in zip(fields[:-1], fields[1:]):
        if (a["ablation_fraction"]-.5)*(b["ablation_fraction"]-.5)<0:
            crossings.append([a["field_mT"],b["field_mT"]])
    summary = {"model":"aligned three-nitrogen full pair; exact mixed-nuclear trace",
        "axial_scaling":"N5, N10 and N1 Azz multiplied together; their ratios preserved",
        "regime_crossover_half_ablation_brackets_mT":crossings,
        "field_at_50uT":fields[0],"field_at_100mT":fields[-1],
        "sensitivity_optima":peaks,"scaling_law":fit,
        "search_boundary_optima":[r for r in peaks if r["lambda"] in (0.,4.)],
        "direct_tau29_peak":direct_peak,"tau29_real_N5_equivalent":direct_real,
        "tau29_real_relative_to_peak":direct_real["gamma_per_rad"]/direct_peak["gamma_per_rad"],
        "numerical_resolution_passed":all(c["passed_0p5_percent"] for c in checks),
        "common_scale_law_passed":fit["passes_5_percent_collapse"] and fit["passes_5_percent_fixed_minus_half_law"],
        "single_nucleus_shape_inference_validated":False,
        "shape_caveat":"A common bath-scale law does not establish a shape law for each individual nucleus. The chemical panel is only conditional.",
        "null_ensemble":"Not recreated: the brief gives sample count 300000 but no prior bounds, distribution or seed."}
    atomic_json(output/"summary.json", summary)
    write_csv(output/"field_metrics.csv", fields)
    write_csv(output/"scaling_metrics.csv", sorted(scaling,key=lambda r:(r["Azz_N5_mT"],r["tau_us"],r["lambda"])))
    write_csv(output/"sensitivity_optima.csv", peaks)
    write_csv(output/"tau29_metrics.csv", direct)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                        "axes.grid":True,"grid.alpha":.18,"pdf.fonttype":42,"ps.fonttype":42})
    colors = ["#287bb5", "#c75383", "#299881"]
    fig, axes = plt.subplots(1,2,figsize=(11.6,4.6),constrained_layout=True)
    B = np.array([r["field_mT"] for r in fields]); f = np.array([r["ablation_fraction"] for r in fields])
    axes[0].plot(B,f,"-o",ms=3,color=colors[0])
    axes[0].axhline(0,color="black",lw=.8);axes[0].axhline(.5,color="gray",ls=":",lw=.8)
    axes[0].set(xscale="log",xlabel=r"field magnitude $B_0$ (mT)",ylabel=r"$f=1-\Gamma_{\rm ablated}/\Gamma_{\rm full}$",title="(a) Ablation of all transverse couplings")
    if f.min() < -5:
        # Retain every field and the signed ratio. Linear around the
        # crossover, logarithmic for large negative blurring ratios.
        axes[0].set_yscale("symlog",linthresh=1,linscale=1)
        axes[0].set_yticks([-1000,-100,-10,-1,0,1],labels=["-1000","-100","-10","-1","0","1"])
        axes[0].text(.02,.92,"y: linear within [-1, 1]; log outside",transform=axes[0].transAxes,fontsize=7.5)
    axes[1].plot(B,[r["theta_full_deg"] for r in fields],"-o",ms=3,color=colors[0],label="full transverse coupling")
    axes[1].plot(B,[r["theta_ablated_deg"] for r in fields],"--",color=colors[1],label="all transverse components zero")
    axes[1].set(xscale="log",xlabel=r"field magnitude $B_0$ (mT)",ylabel=r"maximum-slope orientation $\theta_\Gamma$ (degrees)",title="(b) Location of maximum angular sensitivity",ylim=(-2,92))
    axes[1].legend(fontsize=8,loc="lower right",bbox_to_anchor=(1,.11))
    shade_regimes(axes,fields)
    fig.suptitle("Transverse-to-axial regime change: three-nitrogen toy")
    fig.supxlabel(r"N5 + N10 on FAD, N1 on Trp; aligned axes; $\tau=10\,\mu$s; $J=D=0$"
                  + f"; Q scale={quadrupole_scale:g}; exact nuclear trace",fontsize=9)
    save_figure(fig,output,"three_nitrogen_regime_shift")

    fig,axes = plt.subplots(1,3,figsize=(13.5,4.6),constrained_layout=True)
    for azz,color in zip(AXIAL_N5_MT,colors):
        selected = [r for r in peaks if r["Azz_N5_mT"]==azz]
        t=[r["tau_us"] for r in selected]; b=np.array([r["b_N5_uT"] for r in selected])
        label=rf"$A_{{zz,N5}}={azz:g}$ mT"
        axes[0].plot(t,b,"-o",color=color,label=label)
        axes[1].plot(t,b/azz,"-o",color=color,label=label)
        axes[2].plot(t,b*np.sqrt(t)/azz,"-o",color=color,label=label)
    t=np.geomspace(5,100,200)
    axes[1].plot(t,REFERENCE_C/np.sqrt(t),"k--",lw=1.2,label="original two-nucleus law")
    axes[2].axhline(fit["fixed_minus_half_C"],color="gray",ls=":",label="three-N fixed-exponent fit")
    axes[2].axhline(REFERENCE_C,color="black",ls="--",lw=1.2,label="two-N C = 391")
    for ax in axes:
        ax.set_xscale("log");ax.set_xlabel(r"lifetime $\tau$ ($\mu$s)")
        ax.set_xticks(TAUS,labels=[str(int(t)) for t in TAUS]);ax.legend(fontsize=7.5)
    axes[0].set(yscale="log",ylabel=r"optimal $b_{N5}$ ($\mu$T)",title="(a) Absolute sensitivity optimum")
    axes[1].set(yscale="log",ylabel=r"$b_{N5,\rm opt}/A_{zz,N5}$ ($\mu$T/mT)",title="(b) Test of scale collapse")
    axes[2].set(ylabel=r"$C_{\rm eff}$ ($\mu$T $\sqrt{\mu{\rm s}}$/mT)",title="(c) Test of inverse-square-root law")
    fig.suptitle("Modelling of the transverse-coupling optimum: three nitrogens")
    fig.supxlabel(r"All three $A_{zz}$ scale together; shared transverse scale; $B_0=50\,\mu$T, $J=D=0$"
                  + f"; Q scale={quadrupole_scale:g}",fontsize=9)
    save_figure(fig,output,"three_nitrogen_scaling_law")

    fig,axes = plt.subplots(1,2,figsize=(11.6,4.7),constrained_layout=True)
    axes[0].plot([r["b_N5_uT"] for r in direct],[r["gamma_per_rad"] for r in direct],"-o",color=colors[0],ms=3)
    axes[0].axvline(132.5,color=colors[1],ls="--",label="brief's real-N5 b = 132.5 uT")
    axes[0].scatter(direct_peak["b_N5_uT"],direct_peak["gamma_per_rad"],marker="*",s=130,color=colors[2],edgecolor="black",linewidth=.4,zorder=5,label="three-N optimum")
    axes[0].set(xlabel=r"N5-equivalent $b_{N5}$ ($\mu$T)",ylabel=r"$\Gamma$ (rad$^{-1}$)",title=r"(a) Direct check at $\tau=29\,\mu$s")
    axes[0].legend(fontsize=8)
    axes[0].text(.50,.65,f"132.5 uT: {100*summary['tau29_real_relative_to_peak']:.3f}% of maximum\nOther transverse terms scale with N5.",transform=axes[0].transAxes,fontsize=8.5)
    x=np.linspace(.55,.98,500)
    formula=lambda C:(C/1000*(x+2)/(np.sqrt(2)*abs(x-1)))**2
    axes[1].plot(x,formula(REFERENCE_C),"k--",label="original two-N shape law")
    axes[1].plot(x,formula(fit["fixed_minus_half_C"]),color=colors[0],label="three-N fit: conditional prediction")
    axes[1].fill_between(x,formula(fit["C_eff_min"]),formula(fit["C_eff_max"]),color=colors[0],alpha=.15,label="variation of fitted C (not a CI)")
    for label,aiso,T in (("N5",.5233,.6168),("N1",.3215,.3798),("N10",.1887,.208)):
        ratio=aiso/T
        tau=(REFERENCE_C/1000*(ratio+2)/(math.sqrt(2)*abs(ratio-1)))**2
        axes[1].scatter(ratio,tau,s=25,color="black")
        axes[1].annotate(label,(ratio,tau),xytext=(4,8 if label!="N1" else -14),textcoords="offset points",fontsize=8)
    axes[1].set(yscale="log",xlabel=r"hyperfine shape ratio $x=a_{\rm iso}/T$",ylabel=r"conditional optimal lifetime $\tau^*$ ($\mu$s)",title="(b) Chemical-shape inference diagnostic")
    axes[1].legend(fontsize=7.5,loc="upper left")
    status="is within the 5% scaling-law diagnostic tolerance" if summary["common_scale_law_passed"] else "is NOT within the 5% scaling-law diagnostic tolerance"
    fig.suptitle("Direct optimum check and the limits of shape-only inference")
    fig.supxlabel("Three-N model "+status+"; chemical panel is conditional, not an independent simulation.",fontsize=9)
    save_figure(fig,output,"three_nitrogen_optimum_shape")
    notes = f"""# Three-nitrogen extension of the transverse/axial brief

## Model and accepted mapping

Complete 108-state pair: electron A, electron B, FAD N5, FAD N10, Trp N1.
All three nuclei are spin 1. Principal axes aligned. N5 and N10 couple to A;
N1 couples to B. Unscaled tensors in mT are {NITROGENS!r}.
Exact electronic singlet and maximally mixed nuclear state. Full-pair dense
diagonalization and analytic equal-rate recombination integral; no Hamiltonian
factorization, Monte Carlo uncertainty, quadrature or propagation error.
J=D=0, Q scale={quadrupole_scale:g}, no nuclear Zeeman or relaxation.

## Investigations

1. Field scan: {len(fields)} field magnitudes from 0.05 to 100 mT, tau=10 us.
   Compare lambda=1 with ALL three transverse tensors zeroed. Definition
   f=1-Gamma_ablated/Gamma_full. Plot maximum-slope angle, not feature center.
2. The tau=1,10,100 us coupling/amplitude investigation already completed in
   ../three_nitrogen_optimum is retained unchanged, not rerun.
3. Scaling scan: the original three N5 axial values set a COMMON multiplier
   on ALL THREE Azz values, preserving axial ratios; transverse values share
   their own independent multiplier. Lifetimes 5,10,20,50,100 us, B=50 uT.
   b means the N5-equivalent transverse value, not total nuclear-bath strength.
   Search lambda=0 plus 0.04-4; refine every local maximum at least 90% of the
   largest screened sensitivity, not a prechosen needle branch.
4. Direct check: repeat tau=29 us, include the original nine b values plus a
   broad discovery grid and refined optimum. This scales all three transverse
   terms; 132.5 uT refers only to N5, not the complete physical bath.
5. Shape-ratio interpretation: measured collapse max spread
   {100*max(r['relative_range'] for r in fit['normalized_curve_spread']):.3f}%;
   fixed -1/2-law max residual {100*fit['max_relative_fixed_law_residual']:.3f}%.
   The 5% threshold is a working comparison tolerance, not a statistical test
   or a preregistered criterion from the original brief.
   The chemical figure is explicitly conditional; its shaded band is the
   variation in empirical C, NOT a confidence interval. Individual-nucleus
   x does not specify a three-nucleus tensor shape. Even a successful common
   axial-scale collapse is not proof of independent single-nucleus shape laws.
6. The 300,000-sample null cannot be reproduced from the brief: bounds,
   probability distribution and random seed are not stated. No new prior is
   invented. The proposed future full-bath/Fisher/relaxation studies are not
   part of this reproduction.

## Precision and reproducibility

Atomic checkpoints preserve curves and metrics. Manifest hashes reject
incompatible resume. Coupling optima refined to lambda tolerance 0.0005.
Angular derivative maxima use cubic splines, four separated coarse candidates,
and local refinement; field scans also seed axial degeneracies but still
evaluate only the full-pair Hamiltonian. The 0.5% angular-resolution check
passed: {summary['numerical_resolution_passed']}. See resolution_checks.json.
No stochastic error bars are appropriate for an exact trace calculation.

## Invocation

python "Realistic system size/code/rpm_three_nitrogen_suite.py" --stage all --workers 4 --resume

Source: Paper/Final Results/RPM_transverse_vs_axial_brief_v2.pdf, sections 1-3.
"""
    (output/"experiment_notes.md").write_text(notes,encoding="utf-8")
    print(json.dumps({"crossover_brackets_mT":crossings,"scaling":fit,
        "tau29_peak_b_uT":direct_peak["b_N5_uT"],
        "tau29_real_relative_to_peak":summary["tau29_real_relative_to_peak"],
        "resolution_passed":summary["numerical_resolution_passed"]},indent=2),flush=True)


def ensure_manifest(output, quadrupole_scale=0.):
    configuration={"version":1,"nuclei":NITROGENS,"dimensions":[2,2,3,3,3],
        "axis_convention":"all principal axes aligned", "axial_scan_N5_mT":AXIAL_N5_MT.tolist(),
        "axial_mapping":"all three Azz scaled together relative to original N5 Azz",
        "transverse_mapping":"all three transverse components share lambda",
        "field_grid_mT":field_grid().tolist(),"field_tau_us":10.,
        "scaling_field_mT":.05,"scaling_tau_us":TAUS.tolist(),
        "transverse_grid":transverse_grid().tolist(),"direct_tau_us":DIRECT_TAU,
        "direct_b_uT":DIRECT_B.tolist(),"angular_step_field_deg":.01,
        "angular_step_scaling_deg":.05,"local_step_scaling_deg":.002,
        "local_step_field_deg":"0.001 below 3 mT, 0.0002 above",
        "optimizer_lambda_tolerance":.0005,"J_uT":0.,"D_uT":0.,"quadrupole_scale":quadrupole_scale,
        "initial_state":"singlet times exact maximally mixed nuclear density",
        "yield_method":"complete dense pair Hamiltonian, exact spectral recombination integral"}
    digest=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    path=output/"manifest.json"
    if path.exists():
        if json.loads(path.read_text())["configuration_hash"] != digest:
            raise ValueError("incompatible manifest: choose a new output directory")
    else:
        atomic_json(path,{"configuration":configuration,"configuration_hash":digest,
            "engine_source_hash":hashlib.sha256(Path(__file__).with_name("rpm_three_nitrogen_optimum.py").read_bytes()).hexdigest(),
            "runner_source_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment":{"host":platform.node(),"python":platform.python_version(),
                "numpy":np.__version__,"scipy":scipy.__version__,"threads_per_worker":1}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("all","field","scaling","direct","report"),default="all")
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/three_nitrogen_suite"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    add_quadrupole_argument(parser, default=0.0)
    args=parser.parse_args()
    if args.workers<1:
        parser.error("workers must be positive")
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    q = args.quadrupole_scale
    ensure_manifest(output, q)
    if args.stage in ("all","field"):
        jobs=[(f"field_B{b:.10f}",field_cell,(float(b),.01,1.,q)) for b in field_grid()]
        run_jobs(output,jobs,args.workers,args.resume)
    if args.stage in ("all","scaling"):
        jobs=[(f"scaling_A{a:.8f}_lambda{s:.10f}",scaling_cell,(float(a),float(s),.05,.002,None,"scaling",q))
              for a in AXIAL_N5_MT for s in transverse_grid()]
        run_jobs(output,jobs,args.workers,args.resume)
        refine_all(output,args.workers,quadrupole_scale=q)
    if args.stage in ("all","direct"):
        scales=np.unique(np.r_[transverse_grid(), DIRECT_B/B_N5_AT_ONE_UT])
        jobs=[(f"direct_A1.75690000_lambda{s:.10f}",scaling_cell,
               (1.7569,float(s),.05,.002,[DIRECT_TAU],"direct",q)) for s in scales]
        run_jobs(output,jobs,args.workers,args.resume)
        refine_all(output,args.workers,"direct",q)
    if args.stage in ("all","report"):
        require_complete_inventory(output)
        resolution_checks(output,args.workers,q)
        report(output,q)
    print(f"stage {args.stage} complete: {output}",flush=True)


if __name__=="__main__":
    main()
