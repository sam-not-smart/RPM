"""Exact three-14N quadrupole exploration; laptop only, complete 108-state pair.

Unchanged aligned hyperfine tensors. Nine quadrupole conditions, B=0.05/3 mT,
tau=5/10 us. Exact mixed nuclear trace and analytic recombination integral.
Quadrupole orientations refer to this toy's aligned hyperfine frame, not the
full 14-nucleus molecule. No radical factorization or stochastic trace sampling.
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
from dataclasses import asdict, dataclass
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh

from rpm_three_nitrogen_optimum import PairModel, NITROGENS, embed
from rpm_quantum_needle_model import (TWOPI, GAMMA_E_MHZ_PER_MT,
    QUADRUPOLE_PRINCIPAL_VALUES_MT, spin_operators)
from rpm_statevector_engine import quadrupole_operator, efimova_hore_electron_coupling_operator

FIELDS_MT = (.05, 3.)
TAUS_US = np.array([5., 10.])
TRACE_ANGLES = (0., 90.)
Q_KEYS = ("FAD:N5", "FAD:N10", "TrpH:N1")
BRIDGE_D_UT = -347.875
BRIDGE_J_UT = BRIDGE_D_UT / -3.
BRIDGE_BASE_ANGLES = (0., 80., 85., 88., 89., 90., 91., 92., 95., 100., 180.)
DISPLAY_LABELS = {"off":"quadrupoles off", "n5_only":"N5 only", "n10_only":"N10 only",
    "n1_only":"N1 only", "all_half":"all, 0.5x", "all_nominal":"all, 1x",
    "all_double":"all, 2x", "all_phi45":"all, N5 azimuth 45 deg",
    "all_phi90":"all, N5 azimuth 90 deg"}


@dataclass(frozen=True)
class Condition:
    key: str
    scale: float
    enabled: tuple[bool, bool, bool] = (True, True, True)
    n5_azimuth_deg: float = 0.


CONDITIONS = (
    Condition("off",0.), Condition("n5_only",1.,(True,False,False)),
    Condition("n10_only",1.,(False,True,False)), Condition("n1_only",1.,(False,False,True)),
    Condition("all_half",.5), Condition("all_nominal",1.), Condition("all_double",2.),
    Condition("all_phi45",1.,n5_azimuth_deg=45.),
    Condition("all_phi90",1.,n5_azimuth_deg=90.),
)
BY_KEY = {c.key:c for c in CONDITIONS}


def quadrupole_tensors(condition: Condition) -> tuple[np.ndarray, ...]:
    tensors=[]
    phi=math.radians(condition.n5_azimuth_deg)
    # Columns are Qx, Qy, Qz. Qz remains in the ring plane; Qy is the
    # ring normal. This is a right-handed frame for every azimuth.
    cosine,sine=math.cos(phi),math.sin(phi)
    if condition.n5_azimuth_deg==0.:cosine,sine=1.,0.
    if condition.n5_azimuth_deg==90.:cosine,sine=0.,1.
    rotation=np.column_stack(([-sine,cosine,0.], [0.,0.,1.], [cosine,sine,0.]))
    for i,key in enumerate(Q_KEYS):
        values=np.array(QUADRUPOLE_PRINCIPAL_VALUES_MT[key],dtype=float)
        values-=values.mean()
        tensor=np.diag(values)
        if i==0:
            tensor=rotation@tensor@rotation.T
        tensors.append(tensor*condition.scale*float(condition.enabled[i]))
    return tuple(tensors)


class QuadrupoleModel:
    def __init__(self, field_mT: float, condition: Condition, bridge: bool=False):
        self.base=PairModel("three_nitrogen",field_mT=field_mT)
        self.nuclear_dimension=self.base.nuclear_dimension
        self.dimension=self.base.dimension
        self.tensors_mT=quadrupole_tensors(condition)
        self.extra=np.zeros((self.dimension,self.dimension),dtype=complex)
        for i,tensor in enumerate(self.tensors_mT):
            local=quadrupole_operator(1., TWOPI*GAMMA_E_MHZ_PER_MT*tensor)
            self.extra+=embed(local,i+2,self.base.dims)
        if bridge:
            pair=efimova_hore_electron_coupling_operator(
                BRIDGE_J_UT*.001*GAMMA_E_MHZ_PER_MT,
                BRIDGE_D_UT*.001*GAMMA_E_MHZ_PER_MT,np.array([1.,0.,0.]))
            self.extra+=TWOPI*np.kron(pair,np.eye(self.nuclear_dimension))
        # Use the faster real eigensolver only when every imaginary entry
        # is exactly zero. Never discard the physical imaginary Qxy terms.
        if not np.any(self.extra.imag):
            self.extra=self.extra.real.copy()

    def hamiltonian(self,theta_deg):
        return self.base.hamiltonian(1.,float(theta_deg))+self.extra

    def spectrum(self,theta_deg):
        energies,vectors=eigh(self.hamiltonian(theta_deg),check_finite=False,driver="evd")
        d=self.nuclear_dimension
        singlet_rows=(vectors[d:2*d]-vectors[2*d:3*d])/math.sqrt(2.)
        projector=singlet_rows.conj().T@singlet_rows
        weights=np.abs(projector)**2/d
        return energies,weights

    def yields(self,theta_deg):
        energies,weights=self.spectrum(theta_deg)
        gaps2=(energies[:,None]-energies[None,:])**2
        return np.array([np.sum(weights/(1+tau*tau*gaps2)) for tau in TAUS_US])

    def curve(self,theta):
        return np.array([self.yields(t) for t in theta]).T

    def trace(self,theta_deg,times_us):
        energies,weights=self.spectrum(theta_deg)
        return trace_from_spectrum(energies,weights,times_us)


def trace_from_spectrum(energies,weights,times_us):
    upper=np.triu_indices(len(energies),1)
    gaps=(energies[:,None]-energies[None,:])[upper]
    amplitudes=2*weights[upper]
    constant=float(np.trace(weights))
    trace=np.empty(len(times_us))
    # Bound working memory while retaining every spectral contribution.
    for start in range(0,len(times_us),256):
        block=np.asarray(times_us[start:start+256])
        trace[start:start+256]=constant+np.cos(block[:,None]*gaps[None,:])@amplitudes
    return trace


def angle_grid(field):
    if field<1:
        return np.unique(np.round(np.r_[np.arange(0,180.00001,.05),
            np.arange(85,95.00001,.01)],8))
    return np.round(np.arange(0,180.00001,.01),8)


def extrema_of_spline(theta,y):
    radians=np.deg2rad(theta)
    spline=CubicSpline(radians,y)
    roots=spline.derivative(2).roots(extrapolate=False)
    probes=np.r_[radians,roots]
    gradient=np.abs(spline(probes,1)); index=int(gradient.argmax())
    return float(gradient[index]),float(np.rad2deg(probes[index]))


def candidate_centers(theta,curve,count=4,separation=.15):
    gradient=abs(np.gradient(curve,np.deg2rad(theta),edge_order=2))
    candidates=[]
    for i in np.argsort(gradient)[::-1]:
        t=float(theta[i])
        if all(abs(t-c)>=separation for c in candidates):
            candidates.append(t)
        if len(candidates)==count:
            break
    return candidates


def refine_sensitivities(model,theta,curves,step=.001):
    centers=[]
    for curve in curves:
        centers.extend(candidate_centers(theta,curve))
    intervals=[]
    for center in sorted(set(centers)):
        lo,hi=max(0.,center-.05),min(180.,center+.05)
        if intervals and lo<=intervals[-1][1]:
            intervals[-1][1]=max(hi,intervals[-1][1])
        else:
            intervals.append([lo,hi])
    gamma=np.zeros(2); locations=np.zeros(2)
    for lo,hi in intervals:
        local_t=np.linspace(lo,hi,max(5,int(math.ceil((hi-lo)/step))+1))
        local=model.curve(local_t)
        for i,y in enumerate(local):
            g,a=extrema_of_spline(local_t,y)
            if g>gamma[i]:
                gamma[i],locations[i]=g,a
    return gamma,locations


def angular_cell(field,key):
    started=time.perf_counter(); model=QuadrupoleModel(field,BY_KEY[key])
    theta=angle_grid(field);curves=model.curve(theta)
    gamma,locations=refine_sensitivities(model,theta,curves)
    anchors=model.curve(np.array([0.,85.,90.,95.,180.]))
    rows=[]
    for i,tau in enumerate(TAUS_US):
        rows.append({"field_mT":field,"condition":key,"tau_us":float(tau),
            "phi_0":float(anchors[i,0]),"phi_85":float(anchors[i,1]),
            "phi_90":float(anchors[i,2]),"phi_95":float(anchors[i,3]),
            "phi_180":float(anchors[i,4]),"contrast_0_90":float(anchors[i,0]-anchors[i,2]),
            "needle_shoulders_85_95_minus_90":float((anchors[i,1]+anchors[i,3])/2-anchors[i,2]),
            "angular_range":float(np.ptp(curves[i])),
            "gamma_per_rad":float(gamma[i]),"theta_gamma_deg":float(locations[i]),
            "theta_min_deg":float(theta[curves[i].argmin()]),
            "theta_max_deg":float(theta[curves[i].argmax()])})
    return {"key":f"angular_B{field:g}_{key}","theta_deg":theta,"curves":curves,
            "rows":rows,"elapsed_seconds":time.perf_counter()-started}


def save_cell(output,cell):
    path=output/"checkpoints"/(cell["key"]+".npz")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,theta_deg=cell["theta_deg"],curves=cell["curves"],
        rows_json=json.dumps(cell["rows"]),elapsed_seconds=cell["elapsed_seconds"])
    temporary.replace(path)


def load_cell(output,field,key):
    with np.load(output/"checkpoints"/f"angular_B{field:g}_{key}.npz") as data:
        return data["theta_deg"].copy(),data["curves"].copy(),json.loads(data["rows_json"].item())


def atomic_json(path,data):
    temp=path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8");temp.replace(path)


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def dynamics_cell(field,key,angles,bridge=False):
    start=time.perf_counter();model=QuadrupoleModel(field,BY_KEY[key],bridge)
    times=np.linspace(0,10,8001)
    trace_angles=TRACE_ANGLES if bridge else angles
    traces=np.array([model.trace(t,times) for t in trace_angles])
    yields=model.curve(np.array(angles))
    return {"field_mT":field,"condition":key,"angles_deg":np.array(trace_angles),
        "yield_angles_deg":np.array(angles),
        "times_us":times,"singlet_population":traces,"yields":yields,
        "bridge":bridge,"elapsed_seconds":time.perf_counter()-start}


def save_dynamics(output,cell):
    family="bridge" if cell["bridge"] else "dynamics"
    path=output/"checkpoints"/f"{family}_B{cell['field_mT']:g}_{cell['condition']}.npz"
    temp=path.with_suffix(".tmp.npz")
    np.savez_compressed(temp,**{k:np.asarray(v) for k,v in cell.items()})
    temp.replace(path)


def load_dynamics(output,field,key,bridge=False):
    family="bridge" if bridge else "dynamics"
    with np.load(output/"checkpoints"/f"{family}_B{field:g}_{key}.npz") as data:
        return {k:data[k].copy() for k in data.files}


def run_angular(output,workers,resume):
    pending=[(b,c.key) for b in FIELDS_MT for c in CONDITIONS
        if not(resume and (output/"checkpoints"/f"angular_B{b:g}_{c.key}.npz").exists())]
    print(f"angular sweep: {len(pending)} cells, {workers} single-threaded workers",flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(angular_cell,*args) for args in pending]
        for i,future in enumerate(as_completed(futures),1):
            cell=future.result();save_cell(output,cell)
            print(f"angular [{i}/{len(pending)}] {cell['key']}: {cell['elapsed_seconds']:.1f}s",flush=True)


def selected_angles(output,field):
    theta,off,_=load_cell(output,field,"off")
    _,on,_=load_cell(output,field,"all_nominal")
    maximum=float(theta[np.argmax(abs(on[1]-off[1]))])
    return sorted(set(TRACE_ANGLES+(maximum,)))


def run_dynamics(output,workers,resume):
    pending=[]
    for bridge in (False,True):
        family="bridge" if bridge else "dynamics"
        for field in FIELDS_MT:
            chosen=selected_angles(output,field)
            angles=sorted(set(BRIDGE_BASE_ANGLES+tuple(chosen))) if bridge else chosen
            # All nine conditions are traced in the main toy. The bridge
            # retains only the primary off/on pair at a sparse angle set.
            for key in (("off","all_nominal") if bridge else BY_KEY):
                if not(resume and (output/"checkpoints"/f"{family}_B{field:g}_{key}.npz").exists()):
                    pending.append((field,key,angles,bridge))
    print(f"dynamics/bridge: {len(pending)} cells",flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(dynamics_cell,*args) for args in pending]
        for i,future in enumerate(as_completed(futures),1):
            cell=future.result();save_dynamics(output,cell)
            print(f"traces [{i}/{len(pending)}] B={cell['field_mT']:g}, {cell['condition']}, bridge={cell['bridge']} ({cell['elapsed_seconds']:.1f}s)",flush=True)


def yield_differences(output):
    metrics=[];differences=[]
    for field in FIELDS_MT:
        theta,off,reference=load_cell(output,field,"off")
        for condition in CONDITIONS:
            t,values,rows=load_cell(output,field,condition.key)
            if not np.array_equal(theta,t):
                raise ValueError("nonmatching angle grids")
            for i,row in enumerate(rows):
                delta=values[i]-off[i]
                ref=reference[i]
                updated={**row,"delta_phi_0":row["phi_0"]-ref["phi_0"],
                    "delta_phi_90":row["phi_90"]-ref["phi_90"],
                    "delta_contrast_0_90":row["contrast_0_90"]-ref["contrast_0_90"],
                    "delta_needle":row["needle_shoulders_85_95_minus_90"]-ref["needle_shoulders_85_95_minus_90"],
                    "delta_angular_range":row["angular_range"]-ref["angular_range"],
                    "gamma_ratio_to_off":row["gamma_per_rad"]/ref["gamma_per_rad"],
                    "max_abs_delta_phi":float(abs(delta).max()),
                    "theta_max_abs_delta_phi_deg":float(theta[abs(delta).argmax()]),
                    "mean_delta_phi_over_angle":float(np.trapezoid(delta,np.deg2rad(theta))/np.pi),
                    "max_abs_shape_change_after_0deg_offset":float(abs(delta-delta[0]).max())}
                metrics.append(updated)
                differences.extend({"field_mT":field,"condition":condition.key,"tau_us":float(TAUS_US[i]),
                    "theta_deg":float(a),"phi_s":float(v),"delta_phi_s":float(d)}
                    for a,v,d in zip(theta,values[i],delta))
    return metrics,differences


def run_resolution(output,workers):
    path=output/"resolution_checks.json"
    if path.exists():
        return json.loads(path.read_text())
    checks=[]
    # Refine all conditions' slope maxima at half the final step. Also check
    # midpoint angular samples around the biggest yield change and extrema,
    # rather than claiming that spline interpolation is exact.
    for field in FIELDS_MT:
        for key in BY_KEY:
            theta,curves,rows=load_cell(output,field,key)
            model=QuadrupoleModel(field,BY_KEY[key])
            gamma=np.zeros(2)
            for i,row in enumerate(rows):
                center=row["theta_gamma_deg"]
                fine=np.linspace(max(0.,center-.02),min(180.,center+.02),161)
                values=model.curve(fine)[i]
                gamma[i],_=extrema_of_spline(fine,values)
            changes=[float(abs(gamma[i]/rows[i]["gamma_per_rad"]-1)) for i in range(2)]
            checks.append({"field_mT":field,"condition":key,
                "relative_gamma_changes":changes,"passed_0p5_percent":max(changes)<.005})
    atomic_json(path,checks)
    return checks


def dynamics_metrics(output):
    rows=[]
    for bridge in (False,True):
        for field in FIELDS_MT:
            reference=load_dynamics(output,field,"off",bridge)
            keys=("off","all_nominal") if bridge else BY_KEY
            for key in keys:
                data=load_dynamics(output,field,key,bridge)
                t=data["times_us"]
                for i,angle in enumerate(data["angles_deg"]):
                    j=int(np.flatnonzero(data["yield_angles_deg"]==angle)[0])
                    delta=data["singlet_population"][i]-reference["singlet_population"][i]
                    index=int(abs(delta).argmax())
                    rows.append({"bridge":bridge,"field_mT":field,"condition":key,
                        "theta_deg":float(angle),"max_abs_delta_Ps":float(abs(delta[index])),
                        "time_of_max_delta_us":float(t[index]),
                        "rms_delta_Ps_0_to_10us":float(np.sqrt(np.trapezoid(delta**2,t)/10)),
                        "mean_delta_Ps_0_to_10us":float(np.trapezoid(delta,t)/10),
                        "initial_Ps_error":float(abs(data["singlet_population"][i,0]-1)),
                        "min_Ps":float(data["singlet_population"][i].min()),
                        "max_Ps":float(data["singlet_population"][i].max()),
                        "delta_phi_tau5":float(data["yields"][0,j]-reference["yields"][0,j]),
                        "delta_phi_tau10":float(data["yields"][1,j]-reference["yields"][1,j])})
    return rows


def model_sensitivity_metrics(output):
    rows=[]
    for field in FIELDS_MT:
        theta,off,_=load_cell(output,field,"off")
        nominal=load_cell(output,field,"all_nominal")[1]
        oriented=np.array([load_cell(output,field,key)[1]
            for key in ("all_nominal","all_phi45","all_phi90")])
        single_effects=sum((load_cell(output,field,key)[1]-off
            for key in ("n5_only","n10_only","n1_only")),np.zeros_like(off))
        for i,tau in enumerate(TAUS_US):
            spread=np.ptp(oriented[:,i,:],axis=0)
            residual=nominal[i]-off[i]-single_effects[i]
            rows.append({"field_mT":field,"tau_us":float(tau),
                "max_tested_azimuth_spread_phi":float(spread.max()),
                "angle_max_azimuth_spread_deg":float(theta[spread.argmax()]),
                "max_abs_nonadditive_yield_effect":float(abs(residual).max()),
                "angle_max_nonadditivity_deg":float(theta[abs(residual).argmax()])})
    return rows


def archived_zero_curve_error():
    path=Path(__file__).resolve().parents[2]/"outputs"/"three_nitrogen_optimum"/"checkpoints"/"three_nitrogen_lambda1.0000000000.npz"
    if not path.exists():return None
    with np.load(path) as data:
        ids=np.unique(np.r_[np.arange(0,len(data["theta_deg"]),60),len(data["theta_deg"])-1])
        angles=data["theta_deg"][ids]
        archived=data["curves"][1,ids]
    current=QuadrupoleModel(.05,BY_KEY["off"]).curve(angles)[1]
    return float(abs(current-archived).max())


def ensure_manifest(output):
    config={"schema_version":1,"dimensions":[2,2,3,3,3],"nuclei":NITROGENS,
        "hyperfine_transverse_scale":1.,"hyperfine_axial_scale":1.,
        "fields_mT":FIELDS_MT,"lifetimes_us":TAUS_US.tolist(),"conditions":[asdict(c) for c in CONDITIONS],
        "quadrupole_principal_values_mT":QUADRUPOLE_PRINCIPAL_VALUES_MT,
        "quadrupole_tensors_mT":{c.key:[q.tolist() for q in quadrupole_tensors(c)] for c in CONDITIONS},
        "quadrupole_convention":"H_Q/h=I.Q_MHz.I; code H_Q/hbar=2*pi*28*I.Q_mT.I",
        "quadrupole_orientation":"N5 Qz in xy at specified azimuth, Qy along z; N10/N1 Q axes aligned with HF axes, small tilts omitted",
        "quadrupole_trace_removed":True,"electronic_initial_state":"singlet",
        "nuclear_initial_state":"exact maximally mixed nuclear density, Dn=27",
        "primary_J_uT":0.,"primary_D_uT":0.,"relaxation":False,"nuclear_zeeman":False,
        "bridge_D_paper_uT":BRIDGE_D_UT,"bridge_J_uT":BRIDGE_J_UT,"bridge_axis":[1,0,0],
        "bridge_base_angles_deg":BRIDGE_BASE_ANGLES,
        "low_angle_grid":"0:180:0.05 plus 85:95:0.01 deg",
        "high_angle_grid":"0:180:0.01 deg","local_angle_refinement_deg":.001,
        "trace_times_us":{"start":0.,"stop":10.,"step":.00125},
        "trace_selected_angles":"0,90,max absolute nominal-Q yield difference at tau10; latter selected exploratory",
        "yield_method":"exact complete-pair eigenbasis integral, no time truncation",
        "dynamics_method":"exact complete-pair eigenbasis evolution; no factorization"}
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    path=output/"manifest.json"
    if path.exists():
        if json.loads(path.read_text())["configuration_hash"]!=digest:
            raise ValueError("incompatible output manifest")
    else:
        atomic_json(path,{"configuration_hash":digest,"configuration":config,
            "environment":{"host":platform.node(),"python":platform.python_version(),
                "numpy":np.__version__,"scipy":scipy.__version__,"threads_per_worker":1}})


def require_complete(output):
    missing=[]
    for field in FIELDS_MT:
        for key in BY_KEY:
            for family in ("angular","dynamics"):
                path=output/"checkpoints"/f"{family}_B{field:g}_{key}.npz"
                if not path.exists():missing.append(path.name)
        for key in ("off","all_nominal"):
            path=output/"checkpoints"/f"bridge_B{field:g}_{key}.npz"
            if not path.exists():missing.append(path.name)
    if missing:raise RuntimeError(f"incomplete experiment: {len(missing)} checkpoints missing")


def save_fig(fig,output,name):
    fig.savefig(output/f"{name}.png",dpi=300,facecolor="white")
    fig.savefig(output/f"{name}.pdf",facecolor="white")
    plt.close(fig)


def plot_angular(output,keys,name,title,differences=False,orientation=False):
    colors={"off":"#303030","all_half":"#e69f00","all_nominal":"#287bb5", "all_double":"#ad3376",
        "n5_only":"#d55e00","n10_only":"#009e73","n1_only":"#8b65aa", "all_phi45":"#cc79a7", "all_phi90":"#009e73"}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,field in enumerate(FIELDS_MT):
        t,off,_=load_cell(output,field,"off")
        for column,tau in enumerate(TAUS_US):
            ax=axes[row,column]
            for key in keys:
                _,curves,_=load_cell(output,field,key)
                values=curves[column]-off[column] if differences else curves[column]
                label=DISPLAY_LABELS[key]
                if orientation and key=="all_nominal":label="N5 azimuth 0 deg"
                if orientation and key in ("all_phi45","all_phi90"):label=f"N5 azimuth {BY_KEY[key].n5_azimuth_deg:g} deg"
                ax.plot(t,values,color=colors[key],label=label,lw=1.15)
            if differences:ax.axhline(0,color="gray",lw=.6)
            ax.set(xlabel=r"field orientation $\theta$ (degrees)",
                ylabel=r"$\Delta\Phi_S=\Phi_S(Q)-\Phi_S(0)$" if differences else r"singlet yield $\Phi_S$",
                title=rf"$B_0={field:g}$ mT, $\tau={tau:g}\,\mu$s",xlim=(0,180))
            ax.legend(fontsize=8,loc="best",framealpha=.9)
            if not differences and field<1:
                inset=ax.inset_axes([.57,.10,.39,.35])
                for key in keys:
                    _,curves,_=load_cell(output,field,key)
                    inset.plot(t,curves[column],color=colors[key],lw=.9)
                inset.set_xlim(85,95)
                selected=(t>=85)&(t<=95)
                all_y=np.concatenate([load_cell(output,field,key)[1][column,selected] for key in keys])
                pad=max(1e-5,.08*float(np.ptp(all_y)))
                inset.set_ylim(all_y.min()-pad,all_y.max()+pad)
                inset.tick_params(labelsize=7)
                inset.set_title("central feature",fontsize=8)
    fig.suptitle(title,fontsize=13)
    fig.supxlabel("Aligned three-14N toy; hyperfine tensors unchanged; J=D=0; exact nuclear trace and recombination integral",fontsize=9)
    save_fig(fig,output,name)


def plot_dynamics(output):
    with PdfPages(output/"quadrupole_dynamics.pdf") as pdf:
        for field in FIELDS_MT:
            off=load_dynamics(output,field,"off");on=load_dynamics(output,field,"all_nominal")
            t=off["times_us"];angles=off["angles_deg"]
            fig,axes=plt.subplots(2,len(angles),figsize=(12,6.6),squeeze=False,constrained_layout=True)
            for col,angle in enumerate(angles):
                a=off["singlet_population"][col];b=on["singlet_population"][col]
                axes[0,col].plot(t,a,color="#303030",lw=.55,label="Q off")
                axes[0,col].plot(t,b,color="#287bb5",lw=.55,alpha=.8,label="all Q, nominal")
                axes[1,col].plot(t,b-a,color="#ad3376",lw=.6)
                axes[1,col].axhline(0,color="gray",lw=.6)
                axes[0,col].set_title(rf"$\theta={angle:g}^\circ$"+(" (selected)" if angle not in TRACE_ANGLES else ""))
                axes[0,col].set(ylabel=r"$P_S(t)$",ylim=(0,1.02))
                axes[1,col].set(ylabel=r"$\Delta P_S(t)$",xlabel=r"time ($\mu$s)")
                for ax in axes[:,col]:ax.set_xlim(0,10)
                axes[0,col].legend(fontsize=8)
            fig.suptitle(rf"Quadrupoles change the singlet dynamics: $B_0={field:g}$ mT",fontsize=13)
            fig.supxlabel("Exact ensemble dynamics without survival weighting; lifetimes affect yields, not these normalized P_S(t) traces",fontsize=9)
            fig.savefig(output/f"quadrupole_dynamics_B{field:g}.png",dpi=300,facecolor="white")
            pdf.savefig(fig,facecolor="white");plt.close(fig)


def plot_bridge(output):
    fig,axes=plt.subplots(2,2,figsize=(12,7.4),constrained_layout=True)
    for col,field in enumerate(FIELDS_MT):
        off=load_dynamics(output,field,"off",True);on=load_dynamics(output,field,"all_nominal",True)
        for i,tau in enumerate(TAUS_US):
            axes[0,col].plot(off["yield_angles_deg"],on["yields"][i]-off["yields"][i],"o--",ms=4,
                label=rf"$\tau={tau:g}\,\mu$s; sparse angles",color=("#287bb5","#ad3376")[i])
        axes[0,col].axhline(0,color="gray",lw=.6)
        axes[0,col].set(xlabel=r"field orientation $\theta$ (degrees)",ylabel=r"$\Delta\Phi_S$",
            title=rf"$B_0={field:g}$ mT, exchange/dipole enabled")
        axes[0,col].legend(fontsize=8)
        for angle,color in ((0.,"#287bb5"),(90.,"#ad3376")):
            i=int(np.flatnonzero(off["angles_deg"]==angle)[0])
            axes[1,col].plot(off["times_us"],on["singlet_population"][i]-off["singlet_population"][i],
                color=color,lw=.6,label=rf"$\theta={angle:g}^\circ$")
        axes[1,col].axhline(0,color="gray",lw=.6)
        axes[1,col].set(xlabel=r"time ($\mu$s)",ylabel=r"$\Delta P_S(t)$",xlim=(0,10))
        axes[1,col].legend(fontsize=8)
    fig.suptitle("Small-model bridge to the Kapit Work Station cancellation baseline",fontsize=13)
    fig.supxlabel(r"Three nuclei only; $D_{\rm EH}=-347.875\,\mu$T, $J=115.9583\,\mu$T, $D/J=-3$, dipolar axis x; Q off vs nominal",fontsize=9)
    save_fig(fig,output,"quadrupole_coupling_bridge")


def report(output):
    require_complete(output)
    metrics,differences=yield_differences(output)
    traces=dynamics_metrics(output)
    sensitivities=model_sensitivity_metrics(output)
    checks=json.loads((output/"resolution_checks.json").read_text())
    verification=json.loads((output/"verification.json").read_text())
    archive_error=archived_zero_curve_error()
    verification["archived_q0_curve_max_abs_error"]=archive_error
    verification["passed"]=verification["passed"] and (archive_error is None or archive_error<1e-12)
    atomic_json(output/"verification.json",verification)
    write_csv(output/"angular_metrics.csv",metrics)
    write_csv(output/"angular_yields.csv",differences)
    write_csv(output/"dynamics_metrics.csv",traces)
    write_csv(output/"model_sensitivity_metrics.csv",sensitivities)
    nominal=[r for r in metrics if r["condition"]=="all_nominal"]
    summary={"main_nominal_effects":nominal,
        "model_sensitivity_metrics":sensitivities,
        "nominal_dynamics":[r for r in traces if r["condition"]=="all_nominal" and not r["bridge"]],
        "bridge_nominal_dynamics":[r for r in traces if r["condition"]=="all_nominal" and r["bridge"]],
        "all_numerical_checks_passed":all(r["passed_0p5_percent"] for r in checks) and verification["passed"],
        "max_relative_gamma_resolution_change":max(max(r["relative_gamma_changes"]) for r in checks),
        "interpretation":"Exact toy-model effects, not evidence of the size of effects in the full 14-nucleus bath. sQ=0.5/2 and N5 azimuth variation are sensitivity tests, not uncertainty distributions.",
        "uncertainty":"No sampling confidence bands: exact maximally mixed nuclear trace. Quadrupole strengths/orientations are model assumptions."}
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"analysis_metadata.json",{"reported_UTC":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "source_hashes":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("rpm_three_nitrogen_quadrupole.py","rpm_three_nitrogen_optimum.py","rpm_quantum_needle_model.py")}})
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":.15,"pdf.fonttype":42,"ps.fonttype":42})
    plot_angular(output,["off","all_half","all_nominal","all_double"],
        "quadrupole_strength_yields","Singlet yields with nuclear quadrupole coupling")
    plot_angular(output,["n5_only","n10_only","n1_only","all_nominal"],
        "quadrupole_nucleus_attribution","Which nitrogen changes the singlet yield?",True)
    plot_angular(output,["all_nominal","all_phi45","all_phi90"],
        "quadrupole_orientation_sensitivity","Sensitivity to the uncertain N5 quadrupole azimuth",True,True)
    plot_dynamics(output);plot_bridge(output)
    notes=f"""# Three-nitrogen quadrupole experiment

Approved experiment 1 only. Local laptop calculation; no Kapit Work Station jobs.

## Fixed model

Ordering eA,eB,FAD N5,FAD N10,Trp N1, dimensions 2,2,3,3,3 (108 states).
N5/N10 couple to A, N1 to B; hyperfine values {NITROGENS!r} mT unchanged.
Main study J=D=0. Fields 0.05 and 3 mT; recombination lifetimes 5 and 10 us.
Exact electronic singlet and maximally mixed nuclear trace; no relaxation or
nuclear Zeeman term. Complete pair eigendecomposition, not factorization.

## Quadrupole assumptions

Table 2 of Efimova/Hore (2009), DOI 10.1080/00268970902852624. Principal
values are centralized in rpm_quantum_needle_model.py. Rounded isotropic trace
removed. Code multiplies equivalent-field tensors by 2*pi*28 rad/us per mT.
N5 Qz in the xy plane; Qy along the ring normal z. Its unknown azimuth is
tested at 0,45,90 degrees. N10/N1 axes are provisionally aligned with their
hyperfine axes; the small reported Trp tilt is omitted, not fitted.
Full tensor rhombicity retained. Strength factors 0.5/1/2 are sensitivity
scenarios, not experimental error bars. Unchanged hyperfine lambda=1.

## Observables and scope

All nine conditions: dense complete 0-180 degree angular curves and local
0.001-degree slope refinement. P_S(t) sampled every 0.00125 us from 0-10 us
at 0 and 90 degrees and the largest nominal-Q yield-difference angle at tau10.
The last angle is data-selected and explicitly exploratory. Nuclear trace is
exact; curves have no Monte Carlo sampling confidence interval.

Recombination yields use the analytic spectral integral to infinite time.
The 10-us trace limit therefore imposes NO reaction-yield tail truncation.
Normalized P_S(t) is independent of tau under equal S/T rates; survival-weighted
population would be exp(-t/tau)*P_S(t), and is not mislabeled as P_S here.

Attribution uses each nucleus separately plus all three; contributions need
not add linearly. Delta yield, 0-90 contrast, needle shoulder depth, angular
range and maximum angular slope are reported separately. A mean shift does
not automatically mean altered magnetosensitivity.

The small-model bridge retains only Q off/all nominal and sparse angles with
the literal Efimova-Hore D/J=-3 condition, D=-347.875 uT and J=115.9583 uT,
dipolar axis x. Dashed bridge lines connect sampled points and do not resolve
needle shape. This is still only a three-nucleus model, not the Kapit Work Station study.

## Verification

Complex Hermitian operators and conjugate-transpose singlet projections.
Independent density-matrix exponentiation, quadrupole-only and commuting
axial nulls; archive Q=0 comparisons. All verification passed: {verification['passed']}.
Maximum relative angular-refinement change: {summary['max_relative_gamma_resolution_change']:.4g}.
See verification.json and resolution_checks.json. Exact diagonalization still
has floating-point and angular-resolution errors; no fitted stochastic bands.

## Resume

python "Realistic system size/code/rpm_three_nitrogen_quadrupole.py" --workers 4 --resume

Atomic checkpoints, complete tensors and orientations, configuration hash,
software metadata, raw angular values and raw dynamics are saved alongside
five PDF figures (dynamics has two pages) and white-background PNGs.
"""
    (output/"experiment_notes.md").write_text(notes,encoding="utf-8")
    table=["| Field (mT) | Lifetime (us) | Phi(90), Q off | Phi(90), Q on | Delta Phi(90) | Delta 0-90 contrast |",
           "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in nominal:
        table.append(f"| {row['field_mT']:g} | {row['tau_us']:g} | {row['phi_90']-row['delta_phi_90']:.6f} | {row['phi_90']:.6f} | {row['delta_phi_90']:+.6f} | {row['delta_contrast_0_90']:+.6f} |")
    results="""# Experiment 1 results: quadrupoles in the three-nitrogen toy

Completed on the laptop. No full 14-nucleus or Kapit Work Station run was launched.

## Main result

Nominal quadrupoles clearly change the singlet dynamics and integrated yields
in this model. At 50 uT the central yield dip becomes deeper and broader;
this is an angular-shape change, not only a constant yield offset. N5 gives
the largest individual low-field effect, but the three effects do not add
linearly. At 3 mT, quadrupoles shift and reshape the narrow resonances, and
the detailed result is more sensitive to N5's uncertain in-plane azimuth.

The table uses all three quadrupoles at nominal strength and N5 azimuth 0.
Delta means Q on minus Q off; yields are probabilities, not percentages.

"""+"\n".join(table)+"""

## Scope and robustness

The main table is J=D=0. Hyperfine tensors and their aligned axes are unchanged.
The small-model D/J=-3 bridge also shows a quadrupole effect; it is a sparse
exploration and remains a three-nucleus calculation, not a Kapit Work Station result.
The low-field effect remains across all three tested N5 azimuths; the high-field
fine structure should not be interpreted without its orientation sensitivity.
No inference about the size of the effect in the full bath is made here.

P_S(t) is the normalized exact mixed-nuclear ensemble probability without
survival weighting. Instantaneous differences include phase shifts of coherent
oscillations and should not be equated to a persistent population change.
The 5/10-us lifetimes affect the recombination weighting, not the unitary
Hamiltonian. Both yields are integrated analytically to infinite time.

All tests and numerical checks passed. The archived Q=0 angular values agree
exactly at the sampled comparison points. Independent density-matrix evolution
and commuting/quadrupole-only null controls verify the implementation.
Exact nuclear traces require no stochastic confidence bands; quadrupole
strengths and orientations remain physical model assumptions.

## Figures and raw data

- quadrupole_strength_yields.png / .pdf: off, 0.5x, 1x, 2x.
- quadrupole_nucleus_attribution.png / .pdf: individual nuclei and all three.
- quadrupole_orientation_sensitivity.png / .pdf: N5 azimuths 0,45,90 degrees.
- quadrupole_dynamics.pdf: two pages, with separate PNGs for the two fields.
- quadrupole_coupling_bridge.png / .pdf: sparse D/J=-3 comparison.
- angular_metrics.csv, angular_yields.csv, dynamics_metrics.csv,
  model_sensitivity_metrics.csv, summary.json and checkpoints/ hold the data.
- manifest.json, verification.json, resolution_checks.json and
  experiment_notes.md document the configuration and checks.
"""
    (output/"results_summary.md").write_text(results,encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


def verify_physics(output):
    from scipy.linalg import expm
    model=QuadrupoleModel(.05,BY_KEY["all_phi45"])
    s=np.array([0.,1.,-1.,0.])/math.sqrt(2.)
    projector=np.kron(np.outer(s,s),np.eye(27));rho=projector/27
    h=model.hamiltonian(83.7);times=np.array([0.,.013,.13,1.3,5.,10.])
    actual=model.trace(83.7,times);reference=[]
    for t in times:
        u=expm(-1j*h*t)
        reference.append(float(np.trace(projector@u@rho@u.conj().T).real))
    density_error=float(np.max(abs(actual-reference)))
    base=PairModel("three_nitrogen")
    zero=QuadrupoleModel(.05,BY_KEY["off"])
    q0_error=max(float(np.max(abs(zero.yields(t)-base.curve(1.,np.array([t]),TAUS_US)[:,0]))) for t in (0.,85.,90.,95.,180.))
    # Archive comparison uses independently saved Q=0, tau10 metrics.
    archive=Path(__file__).resolve().parents[2]/"outputs"/"three_nitrogen_suite"/"field_metrics.csv"
    archive_errors=[]
    if archive.exists():
        with archive.open() as stream:
            old=list(csv.DictReader(stream))
        for b in FIELDS_MT:
            for row in old:
                if abs(float(row["field_mT"])-b)<1e-8:
                    theta=float(row["theta_full_deg"])
                    step=.0001
                    angles=np.array([theta-step,theta,theta+step])
                    value=QuadrupoleModel(b,BY_KEY["off"]).curve(angles)[1]
                    gamma=abs((value[2]-value[0])/math.radians(2*step))
                    archive_errors.append(abs(gamma/float(row["gamma_full_per_rad"])-1))
    qonly=model.extra
    only_values=model.base.yields_from_hamiltonian(qonly,TAUS_US)
    quad_only_error=float(np.max(abs(only_values-1)))
    # Pure axial HF and exactly axial, collinear Q must commute, even in
    # a transverse applied field, because Q acts only on the nuclear spins.
    axial=PairModel("three_nitrogen")
    h0=axial.hamiltonian(0.,53.)
    q=np.zeros_like(h0,dtype=complex)
    for i,key in enumerate(Q_KEYS):
        qz=QUADRUPOLE_PRINCIPAL_VALUES_MT[key][2]
        tensor=np.diag([-qz/2,-qz/2,qz])*TWOPI*GAMMA_E_MHZ_PER_MT
        q+=embed(quadrupole_operator(1,tensor),i+2,axial.dims)
    commute_norm=float(np.linalg.norm(q@h0-h0@q))
    commute_error=float(np.max(abs(axial.yields_from_hamiltonian(h0.copy(),TAUS_US)-axial.yields_from_hamiltonian(h0+q,TAUS_US))))
    # Direct numerical integration is only a check; production yields are analytic.
    from scipy.integrate import simpson
    times_check=np.arange(0,120.00001,.000625)
    trace=model.trace(83.7,times_check)
    numerical=np.array([simpson(trace*np.exp(-times_check/tau)/tau,x=times_check) for tau in TAUS_US])
    yield_error=np.abs(numerical-model.yields(83.7))
    data={"density_matrix_trace_max_abs_error":density_error,"q0_engine_max_abs_error":q0_error,
        "q0_archive_relative_gamma_errors":archive_errors,
        "quadrupole_only_no_electron_dynamics_error":quad_only_error,
        "commuting_fixture_commutator_norm":commute_norm,"commuting_fixture_yield_error":commute_error,
        "complex_hamiltonian_max_imaginary_element_rad_per_us":float(abs(h.imag).max()),
        "spectral_vs_Simpson_yield_abs_errors":yield_error.tolist(),
        "Simpson_check_time_stop_us":120.,"Simpson_check_step_us":.000625,
        "Simpson_check_max_omitted_tail":math.exp(-120/10)}
    data["passed"]=bool(density_error<1e-10 and q0_error<1e-12 and quad_only_error<1e-12
        and commute_error<1e-11 and commute_norm<1e-10 and float(yield_error.max())<1e-5
        and max(archive_errors,default=0)<.001 and abs(h.imag).max()>1.)
    atomic_json(output/"verification.json",data)
    if not data["passed"]:raise RuntimeError(f"physics verification failed: {data}")
    print("physics verification passed",flush=True)
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/three_nitrogen_quadrupole"))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--stage",choices=("all","verify","angular","dynamics","report"),default="all")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.workers<1:parser.error("workers must be positive")
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True);ensure_manifest(output)
    if args.stage in ("all","verify"):verify_physics(output)
    if args.stage in ("all","angular"):run_angular(output,args.workers,args.resume)
    if args.stage in ("all","dynamics"):run_dynamics(output,args.workers,args.resume)
    if args.stage in ("all","report"):
        require_complete(output);run_resolution(output,args.workers);report(output)
    print(f"complete: {output}",flush=True)


if __name__=="__main__":main()
