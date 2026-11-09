"""Paired Q-on screen using the original poster and N5 production solvers."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
import scipy

from poster_reproduction.core import (
    PRODUCTION_MODEL, build_hamiltonian,
    make_propagator, random_phase_initial_state,
)
from poster_reproduction.runner import _quantum_cell, cell_path as field_path, planned_tasks
from rpm_nuclear_spin_rotation_control import (
    ControlConfig, atomic_json, atomic_npz, cell_path as pulse_path,
    confirmation_tasks, propagate_control_task,
)
from rpm_quantum_needle_model import QUADRUPOLE_TENSORS_MT, SYSTEM_DIMENSIONS
from rpm_shared_capacity import spare_capacity
from rpm_statevector_engine import SingletPopulationObservable


OFF_FIELD = replace(PRODUCTION_MODEL, samples=4)
ON_FIELD = replace(OFF_FIELD, quadrupole_scale=1.0)
OFF_PULSE = ControlConfig()
ON_PULSE = replace(OFF_PULSE, quadrupole_scale=1.0)


def inventory():
    high = [("field", t) for t in planned_tasks("full", 4, 4096, OFF_FIELD.master_seed, (2,))
            if t.field_mT in (2.0, 10.0) and t.theta_deg in (0.0, 90.0)]
    spin = [("pulse", t) for t in confirmation_tasks(OFF_PULSE, 0.2)
            if t.theta_deg in (85.0, 90.0, 95.0) and t.protocol in ("baseline", "periodic")]
    # Longest jobs first. Keep all eight 10 mT cells on the first wave.
    result = sorted(high + spin, key=lambda row: -(
        17 if row[0] == "field" and row[1].field_mT == 10 else
        8 if row[0] == "pulse" else 7
    ))
    assert len(high) == 16 and len(spin) == 24 and len({t.key for _, t in result}) == 40
    return result


def read_cell(path, task, expected_hash, hash_key, *, allow_screen_reuse=False):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        expected = asdict(task)
        acceptable = [expected]
        # Two original confirmation points reuse the physically identical
        # pilot trajectory; its role tag is bookkeeping, not a model input.
        if allow_screen_reuse and task.sample_id == 0 and task.theta_deg == 90.0:
            acceptable.append({**expected, "role": "screen"})
        if meta["task"] not in acceptable or meta[hash_key] != expected_hash:
            raise ValueError(f"Task or configuration mismatch: {path}")
        y, error = float(data["yield"]), float(data["quadrature_error"])
        if not (math.isfinite(y) and 0 <= y <= 1 and math.isfinite(error) and error >= 0):
            raise ValueError(f"Invalid result: {path}")
        return {"yield": y, "quadrature_error": error, "elapsed_seconds": meta["elapsed_seconds"],
                "recorded_role": meta["task"].get("role")}


def archived(base, tasks):
    results = {}
    for group, task in tasks:
        if group == "field":
            path = field_path(base / "poster_reproduction_multisample", task)
            h, key = OFF_FIELD.configuration_hash, "model_hash"
        else:
            path = pulse_path(base / "nuclear_spin_rotation_control_tau10", task)
            h, key = OFF_PULSE.configuration_hash, "configuration_hash"
        results[task.key] = {
            "checkpoint": str(path), "configuration_hash": h,
            **read_cell(path, task, h, key, allow_screen_reuse=(group == "pulse")),
        }
    return results


def smoke():
    if replace(ON_PULSE.sweep, quadrupole_scale=0.0) != OFF_PULSE.sweep:
        raise RuntimeError("N5 propagation settings changed")
    off = build_hamiltonian(90.0, 10.0, model=OFF_FIELD)
    on = build_hamiltonian(90.0, 10.0, model=ON_FIELD)
    qterms = [t for t in on.terms if t.label.endswith(" quadrupole")]
    other = [t for t in on.terms if not t.label.endswith(" quadrupole")]
    assert {t.label for t in qterms} == {f"{site} quadrupole" for site in QUADRUPOLE_TENSORS_MT}
    assert len(off.terms) == len(other)
    assert all(a.axes == b.axes and a.label == b.label and np.array_equal(a.operator, b.operator)
               for a, b in zip(off.terms, other))
    psi = random_phase_initial_state(SYSTEM_DIMENSIONS, ON_FIELD.master_seed, 0)
    result = make_propagator(on, ON_FIELD).propagate(
        psi, np.array([0.00125, 0.0025]), SingletPopulationObservable(SYSTEM_DIMENSIONS)
    )
    assert np.isfinite(result.observable_values).all()
    assert abs(np.linalg.norm(result.final_state) - 1) < 1e-9
    pulse_task = next(t for g, t in inventory() if g == "pulse" and t.protocol == "periodic")
    short = replace(ON_PULSE, lifetime_us=0.005, survival_tail_bound=math.exp(-1.5))
    data = propagate_control_task(replace(pulse_task, interval_us=0.00375), short)
    assert int(data["pulse_count"]) == 1 and 0 < float(data["yield"]) < 1
    print("Q-on 10 mT and N5 pulse smoke passed", flush=True)


def solve(group, task, output):
    result = _quantum_cell(task, ON_FIELD) if group == "field" else propagate_control_task(task, ON_PULSE)
    path = output / "checkpoints" / f"{task.key}.npz"
    atomic_npz(path, result)
    return float(result["yield"])


def completed(output, group, task):
    path = output / "checkpoints" / f"{task.key}.npz"
    if not path.exists():
        return None
    h, key = ((ON_FIELD.configuration_hash, "model_hash") if group == "field" else
              (ON_PULSE.configuration_hash, "configuration_hash"))
    return read_cell(path, task, h, key)


def report(output, tasks, controls):
    rows = [{"family": g, **asdict(t), "q_off": controls[t.key],
             "q_on": completed(output, g, t)} for g, t in tasks]
    if any(row["q_on"] is None for row in rows):
        raise RuntimeError("Cannot generate final report with missing cells")
    atomic_json(output / "sample_yields.json", rows)
    paired = np.random.default_rng(20261108).integers(0, 4, size=(2000, 4))
    summaries = {}

    def values(group, q, theta, *, field=None, protocol="baseline"):
        matched = sorted((row for row in rows if row["family"] == group and
                          row["theta_deg"] == theta and row["protocol"] == protocol and
                          (field is None or row["field_mT"] == field)), key=lambda row: row["sample_id"])
        assert [row["sample_id"] for row in matched] == list(range(4))
        return np.array([row[q]["yield"] for row in matched])

    def metric(name, off, on):
        delta = on - off
        summaries[name] = {"q_off": float(off.mean()), "q_on": float(on.mean()),
                           "delta": float(delta.mean()), "paired_sample_deltas": delta.tolist(),
                           "paired_bootstrap_95_sampling_only": np.percentile(delta[paired].mean(axis=1), [2.5, 97.5]).tolist()}

    for field in (2.0, 10.0):
        for theta in (0.0, 90.0):
            metric(f"{field:g}mT yield {theta:g}deg",
                   *(values("field", q, theta, field=field) for q in ("q_off", "q_on")))
        metric(f"{field:g}mT contrast 0-90",
               *(values("field", q, 0.0, field=field) - values("field", q, 90.0, field=field)
                 for q in ("q_off", "q_on")))

    def depth(q, protocol):
        return (0.5 * (values("pulse", q, 85.0, protocol=protocol) +
                       values("pulse", q, 95.0, protocol=protocol)) -
                values("pulse", q, 90.0, protocol=protocol))

    for protocol in ("baseline", "periodic"):
        metric(f"N5 {protocol} needle depth", *(depth(q, protocol) for q in ("q_off", "q_on")))
    metric("N5 damping (baseline depth minus pulsed depth)",
           *(depth(q, "baseline") - depth(q, "periodic") for q in ("q_off", "q_on")))
    atomic_json(output / "summary.json", {
        "scope": "Exploratory M=4 paired screen; bootstrap excludes numerical and tensor-orientation uncertainty.",
        "metrics": summaries,
    })


def run(output, tasks, controls, workers):
    todo = [(g, t) for g, t in tasks if completed(output, g, t) is None]
    done = len(tasks) - len(todo)
    print(f"{done}/{len(tasks)} checkpointed; {len(todo)} pending; {workers} workers", flush=True)
    active = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while todo or active:
            slots = min(workers - len(active), len(todo))
            if slots:
                gate = spare_capacity(slots)
                atomic_json(output / "capacity_status.json", gate)
                if gate["allowed"]:
                    for group, task in todo[:slots]:
                        active[pool.submit(solve, group, task, output)] = task.key
                    del todo[:slots]
                else:
                    print(f"Waiting for shared capacity: {gate['reasons']}", flush=True)
            atomic_json(output / "status.json", {"completed": done, "planned": len(tasks),
                        "active": list(active.values()), "queued": len(todo), "failed": False})
            if active:
                finished, _ = wait(active, timeout=60, return_when=FIRST_COMPLETED)
                for f in finished:
                    key = active.pop(f)
                    try:
                        y = f.result()
                    except Exception as exc:
                        atomic_json(output / "failure.json", {"cell": key, "error": repr(exc)})
                        raise
                    done += 1
                    print(f"{done}/{len(tasks)} {key}: Phi={y:.9f}", flush=True)
            elif todo:
                time.sleep(60)
    report(output, tasks, controls)
    atomic_json(output / "status.json", {"completed": done, "planned": len(tasks),
                "active": [], "queued": 0, "failed": False, "report": "summary.json"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("preflight", "run", "smoke"))
    p.add_argument("--archive-base", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--workers", type=int, default=24, choices=range(1, 25))
    args = p.parse_args()
    if args.stage == "smoke":
        smoke()
        return
    if args.archive_base is None or args.output_dir is None:
        p.error("--archive-base and --output-dir are required")
    tasks = inventory()
    controls = archived(args.archive_base, tasks)
    print(f"Matched all {len(controls)} archived Q=0 controls", flush=True)
    if args.stage == "preflight":
        smoke()
        gate = spare_capacity(args.workers)
        print(json.dumps(gate), flush=True)
        if not gate["allowed"]:
            raise RuntimeError("Insufficient shared headroom")
        return
    import fcntl

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {
            "q_on_field": ON_FIELD.stable_dict(), "q_on_n5": ON_PULSE.stable_dict(),
            "q_tensors_mT": {k: v.tolist() for k, v in QUADRUPOLE_TENSORS_MT.items()},
            "q_axis_note": "Original nominal tensors; FAD N5 in-plane azimuth is provisional.",
            "inventory": [{"family": g, "task": asdict(t), "key": t.key} for g, t in tasks],
            "q_off_controls": controls,
            "entry_point_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        manifest = out / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text())["configuration_hash"] != digest:
                raise ValueError("Incompatible configuration; refusing to resume")
        else:
            atomic_json(manifest, {"configuration_hash": digest, "configuration": config,
                        "host": socket.gethostname(), "python": sys.version,
                        "numpy": np.__version__, "scipy": scipy.__version__, "workers_at_start": args.workers,
                        "numerical_threads": 1})
        run(out, tasks, controls, args.workers)


if __name__ == "__main__":
    main()
