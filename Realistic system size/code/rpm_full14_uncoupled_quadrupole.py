"""Paired full-state low-field quadrupole controls with electron J=D=0."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import rpm_full14_quadrupole_experiment as base
from poster_reproduction.core import ProductionModel, _electron_coupling_term
from rpm_quantum_needle_jd_sweeps import efimova_hore_uncoupled_control

np = base.np
ANGLES = tuple(float(a) for a in sorted(set(range(0, 181, 10)) | set(range(85, 96))))


class UncoupledModel(ProductionModel):
    @property
    def condition(self):
        return efimova_hore_uncoupled_control()


@dataclass(frozen=True)
class Config(base.Config):
    angles_deg: tuple[float, ...] = ANGLES

    def model(self, scale):
        return UncoupledModel(master_seed=self.seed, samples=self.samples,
                              lifetime_us=self.lifetime_us, quadrupole_scale=float(scale))

    def stable_dict(self):
        return {**super().stable_dict(), "full_coupled_pair": False,
                "full_pair_state_vector": True, "field_plane": self.model(0).field_plane}


def configuration(**kwargs):
    return Config(angles_deg=ANGLES, **kwargs)


def initial_hash(config, sample):
    nuclei, _ = base.system(config)
    state = base.random_phase_initial_state(base.dimensions_for(nuclei), config.seed, sample)
    return hashlib.sha256(state.tobytes()).hexdigest()


def read_cell(root, task, config):
    path = base.cell_path(root, task)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if (metadata["configuration_hash"] != config.digest
                or metadata["task"] != asdict(task)
                or metadata["initial_state_sha256"] != initial_hash(config, task.sample_id)):
            raise ValueError(f"Incompatible checkpoint: {path}")
        values = {key: float(data[key]) for key in (
            "yield", "quadrature_error", "yield_propagation_estimate", "yield_tail_bound")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Nonfinite checkpoint: {path}")
    return values, metadata


def reuse_completed(source, output, config):
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Reuse and output directories must differ")
    manifest = json.loads((source / "manifest.json").read_text())
    source_config = replace(config, angles_deg=tuple(manifest["configuration"]["angles_deg"]))
    if source_config.digest != manifest["configuration_hash"]:
        raise ValueError("Only the angular inventory may differ when reusing checkpoints")
    wanted = set(base.tasks(config))
    provenance = []
    for task in base.tasks(source_config):
        path, destination = base.cell_path(source, task), base.cell_path(output, task)
        if task not in wanted or not path.exists():
            continue
        read_cell(source, task, source_config)
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if destination.exists():
            read_cell(output, task, config)
        else:
            with np.load(path, allow_pickle=False) as data:
                values = {key: data[key] for key in data.files}
            metadata = json.loads(str(values["metadata_json"]))
            metadata.update(configuration_hash=config.digest,
                            original_configuration_hash=source_config.digest,
                            reused_from=str(path), reuse_source_sha256=source_hash)
            values["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
            base.atomic_npz(destination, values)
        provenance.append({"task": task.key, "source": str(path), "source_sha256": source_hash})
    base.atomic_json(output / "reuse_provenance.json", {
        "count": len(provenance), "source_configuration_hash": source_config.digest,
        "configuration_hash": config.digest, "checkpoints": provenance})
    return len(provenance)


def smoke(config):
    small = replace(config, samples=1, angles_deg=(90.,), lifetime_us=.02,
                    tail_bound=math.exp(-2), state_error_budget=1e-7, smoke=True)
    checks = []
    for task in base.tasks(small):
        model = small.model(task.scale)
        if np.any(_electron_coupling_term(model).operator):
            raise AssertionError("Uncoupled control has nonzero J or D")
        nuclei, radicals = base.system(small)
        dimensions = base.dimensions_for(nuclei)
        h = base.build_hamiltonian(task.theta_deg, small.field_mT, model=model,
                                   nuclei=nuclei, radicals=radicals)
        q_terms = [term for term in h.terms if "quadrupole" in term.label]
        if len(q_terms) != 3 * task.scale:
            raise AssertionError("Wrong nuclear quadrupole terms")
        values = base.propagate(task, small)
        energies, vectors = base.scipy.linalg.eigh(h.explicit_matrix())
        state = base.random_phase_initial_state(dimensions, small.seed, task.sample_id)
        coefficients = vectors.conj().T @ state
        exact = [base.singlet_population(vectors @ (np.exp(-1j * energies * t) * coefficients),
                                       dimensions) for t in values["times_us"]]
        error = float(np.max(np.abs(exact - values["singlet_population"])))
        if error > 1e-6:
            raise AssertionError(f"Dense/full-state mismatch: {error}")
        checks.append({"task": asdict(task), "maximum_Ps_error": error})
    return checks


def report(output, config, coupled=None):
    groups = [("D=J=0", output, config)]
    if coupled is not None:
        manifest = json.loads((coupled / "manifest.json").read_text())
        original = base.Config(**{key: value for key, value in
            manifest["configuration"].items() if key in base.Config.__dataclass_fields__})
        original = replace(original, angles_deg=tuple(original.angles_deg))
        if original.digest != manifest["configuration_hash"]:
            raise ValueError("Coupled archive configuration mismatch")
        a, b = config.stable_dict(), original.stable_dict()
        ignore = {"angles_deg", "coupling", "full_coupled_pair", "full_pair_state_vector", "field_plane"}
        for key in a.keys() & b.keys() - ignore:
            if json.dumps(a[key], sort_keys=True) != json.dumps(b[key], sort_keys=True):
                raise ValueError(f"Unpaired coupled comparison: {key}")
        if not set(config.angles_deg) <= set(original.angles_deg):
            raise ValueError("Coupled comparison lacks requested angles")
        groups.append(("D/J=-3", coupled, original))
    rows, table = [], {}
    for label, root, archive_config in groups:
        for task in base.tasks(config):
            values, meta = read_cell(root, task, archive_config)
            rows.append({"electron_interactions": label, **asdict(task), **values,
                         "checkpoint": str(base.cell_path(root, task)),
                         "initial_state_sha256": meta["initial_state_sha256"]})
            error = sum(values[key] for key in (
                "quadrature_error", "yield_propagation_estimate", "yield_tail_bound"))
            table[label, task.scale, task.theta_deg, task.sample_id] = values["yield"], error
    base._write_csv(output / "raw_sample_yields.csv", rows)
    metrics = []
    for label, _, _ in groups:
        for name, weights in (("0deg_minus_90deg", [(0., 1.), (90., -1.)]),
                ("85deg_95deg_shoulders_minus_90deg", [(85., .5), (95., .5), (90., -1.)])):
            if not {angle for angle, _ in weights} <= set(config.angles_deg):
                continue
            for scales, title in (([(0., 1.)], "Q=0"), ([(1., 1.)], "Q=1"),
                                  ([(1., 1.), (0., -1.)], "Q on minus off")):
                values, errors = [], []
                for sample in range(config.samples):
                    terms = [(sign * weight, table[label, q, angle, sample])
                             for q, sign in scales for angle, weight in weights]
                    values.append(sum(w * y for w, (y, _) in terms))
                    errors.append(sum(abs(w) * e for w, (_, e) in terms))
                lo, hi = base.paired_interval(values, reps=config.bootstrap_replicates)
                error = float(np.mean(errors))
                metrics.append({"metric": name, "comparison": f"{label}, {title}",
                    "M": config.samples, "mean": float(np.mean(values)),
                    "bootstrap_95_low": lo, "bootstrap_95_high": hi,
                    "numerical_allowance": error,
                    "expanded_low": None if lo is None else lo-error,
                    "expanded_high": None if hi is None else hi+error})
    base._write_csv(output / "paired_metrics.csv", metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "run", "report"), default="run")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "outputs/quadrupole tests/full14_uncoupled_quadrupole_tau5_29angles"))
    parser.add_argument("--reuse-dir", type=Path)
    parser.add_argument("--coupled-run", type=Path)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=base.Config.seed)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-wall-hours", type=float, default=0.)
    parser.add_argument("--resource-policy", choices=("exclusive", "spare-capacity"), default="exclusive")
    args = parser.parse_args()
    if args.samples < 1 or args.workers < 1 or args.max_wall_hours < 0:
        parser.error("Samples/workers must be positive; wall time must be nonnegative")
    config = configuration(seed=args.seed, samples=args.samples)
    if args.stage == "smoke":
        print(json.dumps(smoke(config), indent=2))
        return
    output = args.output_dir.resolve()
    if any(path is not None and path.resolve() == output
           for path in (args.reuse_dir, args.coupled_run)):
        parser.error("Archives and output must be different directories")
    lock = base.acquire_run_lock(output)
    try:
        base.ensure_manifest(output, config)
        if args.stage == "run":
            if base.resource_blockers(args.workers, args.resource_policy, output):
                raise RuntimeError("Insufficient safe capacity; no trajectories launched")
            if args.reuse_dir is not None:
                reuse_completed(args.reuse_dir, output, config)
            deadline = time.monotonic()+args.max_wall_hours*3600 if args.max_wall_hours else math.inf
            result = base.run_tasks(base.tasks(config), config, output, args.workers,
                deadline, 0., args.resume or args.reuse_dir is not None,
                "uncoupled_quadrupole_29angles", args.resource_policy)
            base.atomic_json(output / "execution_result.json", result)
            if result["status"] != "complete" or result["completed"] != len(base.tasks(config)):
                print(json.dumps({key: value for key, value in result.items()
                                  if key != "durations_seconds"}))
                return
        report(output, config, args.coupled_run)
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
