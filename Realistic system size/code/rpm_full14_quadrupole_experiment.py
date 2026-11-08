"""Paired low-field quadrupole experiment in the full 14-nucleus radical pair.

Q=0 vs all three paper-based 14N tensors, B=50 uT, tau=5 us, EH D/J=-3.
The complete coupled pair evolves by the existing adaptive Lanczos protocol.
No factorized evolution, retuning, hyperfine ablation or high-field run.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
import scipy.linalg

from poster_reproduction.core import (MASTER_SEED,PRODUCTION_MODEL,build_hamiltonian,
    dimensions_for,random_phase_initial_state)
from rpm_quantum_needle_jd_sweeps import (SweepConfig,coarse_grid_indices,integrate_yield,
    _write_json as atomic_json,_write_csv)
from rpm_quantum_needle_model import (ALL_NUCLEI,FAD_NUCLEI,TRP_NUCLEI,
    NUCLEAR_RADICALS,QUADRUPOLE_TENSORS_MT,SPIN_ORDER)
from rpm_statevector_engine import AdaptiveLanczosPropagator,SingletPopulationObservable,singlet_population

ANGLES=(0.,80.,85.,88.,89.,90.,91.,92.,95.,100.,180.)
PRIORITY=(0.,90.,85.,95.,88.,89.,91.,92.,80.,100.,180.)
SCALES=(0.,1.)


@dataclass(frozen=True)
class Config:
    seed:int=MASTER_SEED
    samples:int=4
    angles_deg:tuple[float,...]=ANGLES
    field_mT:float=.05
    lifetime_us:float=5.
    tail_bound:float=1e-5
    integration_step_us:float=.00125
    coarse_step_us:float=.0025
    state_error_budget:float=2.4e-5
    bootstrap_replicates:int=2000
    smoke:bool=False

    @property
    def sweep(self):
        return replace(SweepConfig(),seed=self.seed,field_microT=self.field_mT*1000,
            lifetime_us=self.lifetime_us,survival_tail_bound=self.tail_bound,
            integration_step_us=self.integration_step_us,coarse_integration_step_us=self.coarse_step_us,
            state_error_budget=self.state_error_budget)

    def model(self,scale):
        return replace(PRODUCTION_MODEL,master_seed=self.seed,samples=self.samples,
            field_plane=PRODUCTION_MODEL.field_plane,lifetime_us=self.lifetime_us,
            quadrupole_scale=float(scale),ratio_D_over_J=-3.)

    def stable_dict(self):
        nuclei,radicals=system(self)
        return {**asdict(self),"schema_version":1,"dimensions":list(dimensions_for(nuclei)),
            "ordering":["eA","eB"]+[f"{r}:{n.name}" for n,r in zip(nuclei,radicals)],
            "full_coupled_pair":True,"factorized_evolution":False,
            "quadrupole_scales":SCALES,
            "quadrupole_tensors_mT":{k:v.tolist() for k,v in QUADRUPOLE_TENSORS_MT.items()},
            "quadrupole_orientation":"existing molecular-frame tensors; N5 in-plane azimuth provisional, not fitted",
            "hyperfine_tensors_mT":{f"{r}:{n.name}":n.tensor_mT.tolist() for n,r in zip(nuclei,radicals)},
            "coupling":{k:v for k,v in self.model(0).condition.stable_dict().items() if k!="quadrupole_scale"},
            "maximum_time_us":self.sweep.maximum_time_us,
            "krylov":{"min_dimension":24,"max_dimension":64,"increment":8,"embedded_offset":4,
                "safety_factor":2.,"initial_step_us":.05,"max_step_us":.25,
                "state_error_budget":self.state_error_budget},
            "initial_state":"electronic singlet times seeded nuclear random phase; common seeds across Q and angle",
            "relaxation":False,"nuclear_Zeeman":False}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.stable_dict(),sort_keys=True).encode()).hexdigest()


def system(config):
    if config.smoke:
        return (FAD_NUCLEI[0],FAD_NUCLEI[1],TRP_NUCLEI[0]),("FAD","FAD","TrpH")
    return ALL_NUCLEI,NUCLEAR_RADICALS


def smoke_config():
    return Config(samples=1,angles_deg=(0.,90.),lifetime_us=.02,
        tail_bound=math.exp(-2),state_error_budget=1e-7,smoke=True)


@dataclass(frozen=True)
class Task:
    scale:float
    theta_deg:float
    sample_id:int

    @property
    def key(self):
        return f"Q{int(self.scale)}_th{self.theta_deg:07.3f}_s{self.sample_id:03d}"


def tasks(config):
    priority={a:i for i,a in enumerate(PRIORITY)}
    return sorted([Task(q,a,s) for s in range(config.samples) for a in config.angles_deg for q in SCALES],
        key=lambda t:(t.sample_id,priority.get(t.theta_deg,999),t.scale))


def pilot_tasks(config):
    return [Task(q,a,0) for a in (0.,90.) for q in SCALES]


def cell_path(output,task):
    return output/"checkpoints"/"cells"/(task.key+".npz")


def atomic_npz(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:np.savez_compressed(stream,**values)
    os.replace(temporary,path)


def make_propagator(h,config,*,probe_horizon=None):
    c=config.sweep
    budget=config.state_error_budget
    if probe_horizon is not None:
        # Match the long-run local error allocation in the short speed test.
        budget*=probe_horizon/c.maximum_time_us
    return AdaptiveLanczosPropagator(h,minimum_dimension=c.lanczos_minimum_dimension,
        maximum_dimension=c.lanczos_maximum_dimension,dimension_increment=c.lanczos_dimension_increment,
        embedded_offset=c.lanczos_embedded_offset,safety_factor=c.lanczos_safety_factor,
        state_error_budget=budget,initial_step_us=c.lanczos_initial_step_us,
        maximum_step_us=c.lanczos_maximum_step_us)


class WallTimeStop(RuntimeError):pass


def propagate(task,config,output=None,deadline=math.inf,probe_horizon=None):
    start=time.monotonic();nuclei,radicals=system(config);dimensions=dimensions_for(nuclei)
    h=build_hamiltonian(task.theta_deg,config.field_mT,model=config.model(task.scale),
        nuclei=nuclei,radicals=radicals)
    initial=random_phase_initial_state(dimensions,config.seed,task.sample_id)
    construction=time.monotonic()-start
    horizon=config.sweep.maximum_time_us if probe_horizon is None else probe_horizon
    times=config.sweep.time_grid_us if probe_horizon is None else np.array([0.,probe_horizon])
    last_progress=[0.]
    def progress(t,accepted,rejected,matvecs,error):
        now=time.monotonic()
        if now>=deadline:raise WallTimeStop("wall-time gate reached inside trajectory")
        if output is not None and now-last_progress[0]>=60:
            atomic_json(output/"progress"/(task.key+".json"),{
                "task":asdict(task),"pid":os.getpid(),"time_us":t,"target_time_us":horizon,
                "accepted_steps":accepted,"rejected_steps":rejected,"matvecs":matvecs,
                "accumulated_state_error_estimate":error,"elapsed_seconds":now-start,
                "updated_UTC":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
            last_progress[0]=now
    result=make_propagator(h,config,probe_horizon=probe_horizon).propagate(
        initial,times,SingletPopulationObservable(dimensions),progress_callback=progress)
    elapsed=time.monotonic()-start
    if probe_horizon is not None:
        return {"elapsed_seconds":elapsed,"propagation_seconds":elapsed-construction,
            "simulated_us":probe_horizon,"max_norm_error":result.stats.maximum_norm_error}
    fine=integrate_yield(times,result.observable_values,config.lifetime_us)
    ids=coarse_grid_indices(config.sweep)
    coarse=integrate_yield(times[ids],result.observable_values[ids],config.lifetime_us)
    weights=np.exp(-times/config.lifetime_us)/config.lifetime_us
    propagation_estimate=float(scipy.integrate.simpson(weights*result.observable_error_bounds,x=times))
    metadata={"task":asdict(task),"configuration_hash":config.digest,"elapsed_seconds":elapsed,
        "construction_seconds":construction,"host":socket.gethostname(),"pid":os.getpid(),
        "initial_state_sha256":hashlib.sha256(initial.tobytes()).hexdigest(),
        "stats":asdict(result.stats)}
    return {"yield":np.array(fine),"coarse_yield":np.array(coarse),
        "quadrature_error":np.array(abs(fine-coarse)),"yield_propagation_estimate":np.array(propagation_estimate),
        "yield_tail_bound":np.array(config.tail_bound),"times_us":times,
        "singlet_population":result.observable_values,"population_error_estimate":result.observable_error_bounds,
        "state_error_estimates":result.state_error_estimates,"final_state":result.final_state,
        "initial_singlet":np.array(singlet_population(initial,dimensions)),
        "metadata_json":np.array(json.dumps(metadata,sort_keys=True))}


def other_compute():
    """Read-only, conservative shared-machine guard; never modifies processes."""
    if platform.system()!="Linux":return []
    raw=subprocess.check_output(["ps","-eo","user=,pid=,ppid=,pcpu=,rss=,comm=,args="],text=True)
    rows=[]
    for line in raw.splitlines():
        parts=line.split(None,6)
        if len(parts)<7:continue
        user,pid,ppid,cpu,rss,command,args=parts
        rows.append({"user":user,"pid":int(pid),"ppid":int(ppid),"cpu_percent":float(cpu),
            "rss_GiB":int(rss)/1024**2,"command":command,"args":args})
    own={os.getpid()}
    for _ in range(12):
        added={r["pid"] for r in rows if r["ppid"] in own};before=len(own);own.update(added)
        if len(own)==before:break
    busy=[]
    for r in rows:
        if r["pid"] in own:continue
        scientific=any(name in r["command"].lower() for name in ("python","julia","matlab","rscript","mpirun","lammps","gromacs"))
        if r["cpu_percent"]>=150 or r["rss_GiB"]>=4 or (scientific and r["cpu_percent"]>=30):
            busy.append(r)
    return busy


def resource_blockers(workers,policy="exclusive",output=None):
    if policy=="exclusive":return other_compute()
    if policy!="spare-capacity":raise ValueError("unknown resource policy")
    from rpm_shared_capacity import spare_capacity
    snapshot=spare_capacity(workers)
    if output is not None:atomic_json(output/"capacity_status.json",snapshot)
    return [] if snapshot["allowed"] else [{"reason":"insufficient_spare_capacity","capacity":snapshot}]


def ensure_manifest(output,config):
    output.mkdir(parents=True,exist_ok=True);path=output/"manifest.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old["configuration_hash"]!=config.digest:raise ValueError("incompatible resume manifest")
        return
    files=[Path(__file__),Path(__file__).with_name("rpm_statevector_engine.py"),
        Path(__file__).with_name("rpm_quantum_needle_model.py"),
        Path(__file__).parent/"poster_reproduction"/"core.py"]
    atomic_json(path,{"experiment":"full14_lowfield_quadrupole_off_on",
        "configuration_hash":config.digest,"configuration":config.stable_dict(),
        "created_UTC":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "environment":{"host":socket.gethostname(),"python":platform.python_version(),
            "numpy":np.__version__,"scipy":scipy.__version__,"threads_per_process":1},
        "source_sha256":{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    _write_csv(output/"task_inventory.csv",[{**asdict(t),"key":t.key} for t in tasks(config)])


def count_completed(output,config):return sum(cell_path(output,t).exists() for t in tasks(config))


def remaining_wall_hours(deadline):
    """JSON-safe remaining allocation; None means no scheduled wall-time stop."""
    return None if math.isinf(deadline) else max(0.,(deadline-time.monotonic())/3600)


def run_tasks(todo,config,output,workers,deadline,start_guard_seconds,resume,stage,resource_policy="exclusive"):
    pending=[];durations=[]
    for task in todo:
        if cell_path(output,task).exists():
            if not resume:raise FileExistsError("pass --resume for existing checkpoints")
            with np.load(cell_path(output,task)) as data:
                metadata=json.loads(str(data["metadata_json"]))
                if metadata["configuration_hash"]!=config.digest:
                    raise ValueError("checkpoint has incompatible configuration")
                durations.append(float(metadata["elapsed_seconds"]))
        else:pending.append(task)
    active={};reason="complete";last_resource=0.;busy=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            now=time.monotonic()
            if now-last_resource>60:
                busy=resource_blockers(workers,resource_policy,output);last_resource=now
                if busy and resource_policy=="exclusive":
                    reason="competing_work_detected";atomic_json(output/"competing_work.json",busy)
            guard=max(start_guard_seconds,1.5*max(durations,default=0))
            if now+guard>=deadline and pending and reason=="complete":reason="wall_time_checkpoint"
            while pending and len(active)<workers and reason=="complete" and not busy:
                task=pending.pop(0)
                active[pool.submit(propagate,task,config,output,deadline)]=task
            if not active:
                if pending and busy and resource_policy=="spare-capacity" and reason=="complete":
                    atomic_json(output/"status.json",{"status":"waiting_for_spare_capacity","stage":stage,
                        "completed":count_completed(output,config),"planned":len(tasks(config)),
                        "workers":workers,"resource_policy":resource_policy,
                        "wall_time_limit_enabled":math.isfinite(deadline),
                        "remaining_wall_hours":remaining_wall_hours(deadline)})
                    time.sleep(min(30.,max(0.,deadline-time.monotonic())))
                    continue
                break
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                task=active.pop(future)
                try:values=future.result()
                except WallTimeStop:
                    reason="wall_time_checkpoint";continue
                except Exception as exc:
                    reason="trajectory_failure"
                    atomic_json(output/"failures"/(task.key+".json"),{
                        "task":asdict(task),"error":repr(exc)})
                    print(f"FAILED {task.key}: {exc!r}; no further cells will start",flush=True)
                    continue
                atomic_npz(cell_path(output,task),values)
                metadata=json.loads(str(values["metadata_json"]))
                durations.append(float(metadata["elapsed_seconds"]))
                progress_path=output/"progress"/(task.key+".json")
                atomic_json(progress_path,{"task":asdict(task),"status":"complete",
                    "elapsed_seconds":metadata["elapsed_seconds"],"time_us":config.sweep.maximum_time_us})
                print(f"complete {task.key}: Phi={float(values['yield']):.9f}, {durations[-1]/3600:.3f} h",flush=True)
            atomic_json(output/"status.json",{"status":"running" if active else reason,"stage":stage,
                "completed":count_completed(output,config),"planned":len(tasks(config)),
                "active":[asdict(t) for t in active.values()],"unstarted_in_stage":len(pending),
                "workers":workers,"updated_UTC":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                "resource_policy":resource_policy,"new_starts_held_for_capacity":bool(busy),
                "wall_time_limit_enabled":math.isfinite(deadline),
                "remaining_wall_hours":remaining_wall_hours(deadline)})
    return {"status":reason,"pending":len(pending),"durations_seconds":durations,
        "completed":count_completed(output,config)}


def benchmark(config,output,max_workers,resource_policy="exclusive"):
    path=output/"worker_benchmark.json"
    if path.exists():
        saved=json.loads(path.read_text())
        if saved["max_worker_cap"]==max_workers:return saved
    rows=[]
    for n in (1,2,4,8,16):
        if n>max_workers:continue
        busy=resource_blockers(n,resource_policy,output)
        if busy:raise RuntimeError(f"competing work prevents benchmark: {busy}")
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=n) as pool:
            futures=[pool.submit(propagate,Task(1.,90.,i),config,None,math.inf,.1) for i in range(n)]
            values=[f.result() for f in futures]
        elapsed=time.monotonic()-start
        rows.append({"workers":n,"elapsed_seconds":elapsed,"throughput_us_per_second":.1*n/elapsed,
            "median_cell_seconds":float(np.median([v["elapsed_seconds"] for v in values]))})
        print(f"benchmark {n} workers: {rows[-1]['throughput_us_per_second']:.5g} simulated us/s",flush=True)
    if not rows:raise ValueError("at least one worker required")
    best=max(r["throughput_us_per_second"] for r in rows)
    selected=min(r["workers"] for r in rows if r["throughput_us_per_second"]>=.95*best)
    result={"rows":rows,"selected_workers":selected,"max_worker_cap":max_workers,
        "note":"Short prefix with long-run error-per-time allocation; only worker selection, not an absolute runtime guarantee."}
    atomic_json(path,result);return result


def pilot_projection(output,config,workers,bench):
    durations=[]
    for task in pilot_tasks(config):
        with np.load(cell_path(output,task)) as data:
            durations.append(json.loads(str(data["metadata_json"]))["elapsed_seconds"])
    pilot_workers=min(4,workers)
    performance={r["workers"]:r for r in bench["rows"]}
    reference=performance.get(pilot_workers,performance[min(performance)])
    chosen=performance[workers]
    contention=max(1.,chosen["median_cell_seconds"]/reference["median_cell_seconds"])
    remaining=len(tasks(config))-count_completed(output,config)
    estimate=math.ceil(remaining/workers)*max(durations)*contention*1.25
    result={"pilot_cell_seconds":durations,"selected_workers":workers,"remaining_cells":remaining,
        "short_benchmark_contention_factor":contention,"safety_factor":1.25,
        "projected_remaining_hours":estimate/3600,"provisional":True}
    atomic_json(output/"runtime_projection.json",result);return result


def paired_interval(values,seed=1701,reps=2000):
    values=np.asarray(values,float)
    if len(values)<2:return [None,None]
    ids=np.random.default_rng(seed).integers(0,len(values),(reps,len(values)))
    return np.quantile(values[ids].mean(axis=1),[.025,.975]).tolist()


def trace_report(output,config):
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,constrained_layout=True)
    any_data=False;summary=[]
    for column,angle in enumerate((0.,90.)):
        available=[s for s in range(config.samples) if all(cell_path(output,Task(q,angle,s)).exists() for q in SCALES)]
        if not available:continue
        any_data=True;traces={};errors={};times=None
        for q in SCALES:
            y=[];e=[]
            for s in available:
                with np.load(cell_path(output,Task(q,angle,s))) as data:
                    t=data["times_us"];keep=t<=10.+1e-12
                    times=t[keep];y.append(data["singlet_population"][keep]);e.append(data["population_error_estimate"][keep])
            traces[q]=np.array(y);errors[q]=np.array(e)
        off=traces[0.].mean(axis=0);on=traces[1.].mean(axis=0)
        delta=traces[1.]-traces[0.];mean=delta.mean(axis=0)
        axes[0,column].plot(times,off,lw=.9,label="Q off")
        axes[0,column].plot(times,on,lw=.9,label="Q on")
        axes[1,column].plot(times,mean,color="#ad3376",lw=.9)
        if len(available)>=2:
            ids=np.random.default_rng(1701).integers(0,len(available),(config.bootstrap_replicates,len(available)))
            counts=np.array([(ids==i).sum(axis=1) for i in range(len(available))]).T/len(available)
            low=np.empty(len(times));high=low.copy()
            numeric=(errors[0.]+errors[1.]).mean(axis=0)
            for start in range(0,len(times),1000):
                draws=counts@delta[:,start:start+1000]
                lower,upper=np.quantile(draws,[.025,.975],axis=0)
                low[start:start+1000]=lower-numeric[start:start+1000]
                high[start:start+1000]=upper+numeric[start:start+1000]
            axes[1,column].fill_between(times,low,high,color="#ad3376",alpha=.2)
        averages=np.array([scipy.integrate.simpson(d,x=times)/(times[-1]-times[0]) for d in delta])
        summary.append({"theta_deg":angle,"paired_samples":len(available),
            "trace_stop_us":float(times[-1]),"mean_delta_Ps_time_average":float(averages.mean()),
            "time_average_stochastic_interval":paired_interval(averages,reps=config.bootstrap_replicates),
            "max_abs_mean_delta_Ps_descriptive":float(abs(mean).max())})
        axes[0,column].set_title(f"theta={angle:g} deg; M={len(available)} paired")
        axes[0,column].set_ylabel("normalized P_S(t)");axes[0,column].legend()
        axes[1,column].axhline(0,color="gray",lw=.6)
        axes[1,column].set(xlabel="time (us)",ylabel="Q-on minus Q-off P_S(t)")
        for ax in axes[:,column]:ax.grid(alpha=.2)
    if any_data:
        fig.suptitle("Full-pair singlet dynamics; 50 uT, EH D/J=-3\nPointwise exploratory paired intervals; not a simultaneous band")
        fig.savefig(output/"singlet_dynamics_comparison.png",dpi=220,facecolor="white")
    plt.close(fig)
    atomic_json(output/"dynamics_summary.json",summary)


def report(output,config):
    raw=[];lookup={}
    for task in tasks(config):
        path=cell_path(output,task)
        if not path.exists():continue
        with np.load(path) as data:
            m=json.loads(str(data["metadata_json"]))
            row={**asdict(task),"yield":float(data["yield"]),"quadrature_error":float(data["quadrature_error"]),
                "propagation_estimate":float(data["yield_propagation_estimate"]),
                "tail_bound":float(data["yield_tail_bound"]),"elapsed_seconds":m["elapsed_seconds"]}
            raw.append(row);lookup[(task.sample_id,task.theta_deg,task.scale)]=row
    _write_csv(output/"raw_sample_yields.csv",raw)
    rows=[]
    for angle in sorted(config.angles_deg):
        paired=[s for s in range(config.samples) if all((s,angle,q) in lookup for q in SCALES)]
        if not paired:continue
        deltas=np.array([lookup[(s,angle,1.)]["yield"]-lookup[(s,angle,0.)]["yield"] for s in paired])
        systematic=np.mean([sum(lookup[(s,angle,q)]["propagation_estimate"]+
            lookup[(s,angle,q)]["quadrature_error"]+lookup[(s,angle,q)]["tail_bound"] for q in SCALES) for s in paired])
        ci=paired_interval(deltas,reps=config.bootstrap_replicates)
        rows.append({"theta_deg":angle,"paired_samples":len(paired),
            "mean_q_off":float(np.mean([lookup[(s,angle,0.)]["yield"] for s in paired])),
            "mean_q_on":float(np.mean([lookup[(s,angle,1.)]["yield"] for s in paired])),
            "mean_delta_yield":float(deltas.mean()),"stochastic_low":ci[0],"stochastic_high":ci[1],
            "numerical_halfwidth_estimate":float(systematic),
            "total_low":None if ci[0] is None else ci[0]-systematic,
            "total_high":None if ci[1] is None else ci[1]+systematic})
    _write_csv(output/"paired_angle_results.csv",rows)
    metrics=[]
    for label,angles,weights in (("contrast_0_minus_90",(0.,90.),(1.,-1.)),
        ("needle_shoulders_minus_90",(85.,90.,95.),(.5,-1.,.5))):
        samples=[s for s in range(config.samples) if all((s,a,q) in lookup for a in angles for q in SCALES)]
        if not samples:continue
        off=np.array([sum(w*lookup[(s,a,0.)]["yield"] for a,w in zip(angles,weights)) for s in samples])
        on=np.array([sum(w*lookup[(s,a,1.)]["yield"] for a,w in zip(angles,weights)) for s in samples])
        error=float(np.mean([sum(abs(w)*sum(lookup[(s,a,q)]["propagation_estimate"]+
            lookup[(s,a,q)]["quadrature_error"]+lookup[(s,a,q)]["tail_bound"] for q in SCALES)
            for a,w in zip(angles,weights)) for s in samples]))
        low,high=paired_interval(on-off,reps=config.bootstrap_replicates)
        metrics.append({"metric":label,"paired_samples":len(samples),"q_off":float(off.mean()),
            "q_on":float(on.mean()),"delta":float((on-off).mean()),"stochastic_low":low,
            "stochastic_high":high,"numerical_halfwidth_estimate":error,
            "total_low":None if low is None else low-error,"total_high":None if high is None else high+error})
    _write_csv(output/"paired_angular_metrics.csv",metrics)
    trace_report(output,config)
    complete=len(raw)==len(tasks(config))
    atomic_json(output/"analysis_summary.json",{"complete":complete,"completed":len(raw),
        "planned":len(tasks(config)),"angle_results":rows,"angular_metrics":metrics,
        "confidence_note":"Paired nuclear-row bootstrap, exploratory M=4 intervals; numerical estimates added conservatively. M=1 has no sampling interval. No equivalence/null conclusion from a non-resolved effect."})
    if rows:
        fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True,constrained_layout=True)
        t=np.array([r["theta_deg"] for r in rows])
        axes[0].plot(t,[r["mean_q_off"] for r in rows],"o-",label="quadrupoles off")
        axes[0].plot(t,[r["mean_q_on"] for r in rows],"o-",label="all three quadrupoles on")
        axes[0].set_ylabel("singlet yield");axes[0].legend()
        axes[1].plot(t,[r["mean_delta_yield"] for r in rows],"o-",color="#ad3376")
        if all(r["total_low"] is not None for r in rows):
            axes[1].fill_between(t,[r["total_low"] for r in rows],[r["total_high"] for r in rows],alpha=.2,color="#ad3376")
        axes[1].axhline(0,color="gray",lw=.7);axes[1].set(xlabel="field orientation (degrees)",ylabel="Q-on minus Q-off yield")
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle("Full 14-nucleus quadrupole comparison"+("" if complete else " (PARTIAL)")+"\n50 uT, tau=5 us, EH D/J=-3")
        fig.savefig(output/"angular_quadrupole_comparison.png",dpi=220,facecolor="white");plt.close(fig)
    (output/"report.md").write_text(f"""# Full-bath low-field quadrupole experiment

Completed trajectories: {len(raw)}/{len(tasks(config))}. Complete: {complete}.
50 uT, tau=5 us, 14 physical nuclei, full coupled state of dimension 221184.
EH D=-347.875 uT, J=115.958333 uT, D/J=-3; dipolar axis x.
All hyperfines unchanged. Q off vs all three existing paper-based tensors.
N5's in-plane quadrupole azimuth is provisional and fixed, not optimized.
M=4 seeded nuclear phase samples paired across Q and angles.
No factorized propagation; unchanged adaptive Lanczos tolerances.
Simpson 0.00125-us grid checked against 0.0025-us decimation.
Tail bound 1e-5; maximum trajectory time {config.sweep.maximum_time_us:.8f} us.
Final states, complete P_S(t), errors and runtime metadata remain in checkpoints.
Paired bootstrap nuclear intervals and numerical estimates are separated.
Small-M intervals are exploratory; unresolved differences do not establish no effect.
Partial plots are explicitly labeled and must not be presented as complete M=4 curves.
""",encoding="utf-8")


def smoke(output):
    config=smoke_config();ensure_manifest(output,config);checks=[]
    for task in pilot_tasks(config):
        data=propagate(task,config,output)
        nuclei,radicals=system(config);dims=dimensions_for(nuclei)
        h=build_hamiltonian(task.theta_deg,config.field_mT,model=config.model(task.scale),
            nuclei=nuclei,radicals=radicals).explicit_matrix()
        energies,vectors=scipy.linalg.eigh(h)
        initial=random_phase_initial_state(dims,config.seed,task.sample_id)
        coefficients=vectors.conj().T@initial
        reference=np.array([singlet_population(vectors@(np.exp(-1j*energies*t)*coefficients),dims) for t in data["times_us"]])
        error=float(abs(reference-data["singlet_population"]).max())
        if error>1e-6:raise RuntimeError(f"dense/Krylov smoke failed: {error}")
        atomic_npz(cell_path(output,task),data);checks.append({"task":asdict(task),"max_Ps_error":error})
    atomic_json(output/"smoke_checks.json",{"passed":True,"checks":checks})
    print("dense/Krylov three-nitrogen smoke PASSED",flush=True)


def acquire_run_lock(output):
    if platform.system()!="Linux":return None
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    handle=(output/"run.lock").open("a+",encoding="utf-8")
    try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close();raise RuntimeError("another process already owns this experiment output")
    handle.seek(0);handle.truncate()
    json.dump({"pid":os.getpid(),"host":socket.gethostname(),"started_unix":time.time()},handle)
    handle.flush()
    return handle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("auto","pilot","full","report","smoke","benchmark"),default="auto")
    p.add_argument("--output-dir",type=Path,default=Path("outputs/full14_quadrupole_lowfield_tau5"))
    p.add_argument("--max-workers",type=int,default=16)
    p.add_argument("--max-wall-hours",type=float,default=24.)
    p.add_argument("--no-wall-limit",action="store_true",
        help="Run until the requested inventory is finished, without a scheduled wall-time stop; retain atomic result saves.")
    p.add_argument("--start-guard-hours",type=float,default=1.)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--resource-policy",choices=("exclusive","spare-capacity"),default="exclusive")
    args=p.parse_args()
    if args.max_workers<1 or args.max_workers>16:p.error("worker cap must be between 1 and 16")
    if (not math.isfinite(args.max_wall_hours) or args.max_wall_hours<=0 or
            not math.isfinite(args.start_guard_hours) or args.start_guard_hours<0):
        p.error("invalid wall-time budget")
    output=args.output_dir
    if args.stage=="smoke":smoke(output);return
    config=Config();ensure_manifest(output,config)
    deadline=math.inf if args.no_wall_limit else time.monotonic()+args.max_wall_hours*3600
    if args.stage=="report":report(output,config);return
    # Keep the descriptor alive for this complete run. Linux flock is
    # released automatically on return, exceptions or process termination.
    run_lock=acquire_run_lock(output)
    atomic_json(output/("execution_"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+".json"),{
        "pid":os.getpid(),"host":socket.gethostname(),"max_workers":args.max_workers,
        "resource_policy":args.resource_policy,"stage":args.stage,
        "wall_time_limit_enabled":not args.no_wall_limit,
        "max_wall_hours":None if args.no_wall_limit else args.max_wall_hours,
        "resume":args.resume,"runner_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration_hash":config.digest})
    busy=resource_blockers(args.max_workers,args.resource_policy,output)
    if busy:
        atomic_json(output/"competing_work.json",busy)
        atomic_json(output/"status.json",{"status":"competing_work_detected","processes":busy})
        print("New work detected; no simulation launched.",flush=True);return
    bench=benchmark(config,output,args.max_workers,args.resource_policy);workers=bench["selected_workers"]
    if args.stage=="benchmark":return
    if args.stage in ("auto","pilot"):
        result=run_tasks(pilot_tasks(config),config,output,min(4,workers),deadline,
            args.start_guard_hours*3600,args.resume,"pilot",args.resource_policy)
        if result["status"]!="complete":report(output,config);return
        projection=pilot_projection(output,config,workers,bench)
        remaining=(deadline-time.monotonic())/3600
        if args.stage=="pilot" or projection["projected_remaining_hours"]>remaining:
            atomic_json(output/"status.json",{"status":"pilot_complete_runtime_gate_pending",
                "completed":count_completed(output,config),"planned":len(tasks(config)),
                "projection":projection,"remaining_allocation_hours":remaining})
            report(output,config);print("Pilot complete; full sweep held at runtime gate.",flush=True);return
    result=run_tasks(tasks(config),config,output,workers,deadline,args.start_guard_hours*3600,
        args.resume,"full",args.resource_policy)
    report(output,config)
    atomic_json(output/"status.json",{"status":"complete" if count_completed(output,config)==len(tasks(config)) else result["status"],
        "completed":count_completed(output,config),"planned":len(tasks(config)),"workers":workers,
        "wall_time_limit_enabled":not args.no_wall_limit})


if __name__=="__main__":main()
