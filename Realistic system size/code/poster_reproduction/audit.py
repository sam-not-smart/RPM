"""Audit poster-reproduction checkpoints and final deliverables."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .core import MASTER_SEED, PRODUCTION_MODEL, atomic_json
from .runner import (
    DEFAULT_OUTPUT,
    cell_path,
    planned_refinement_tasks,
    planned_tasks,
)


REQUIRED_TABLES = (
    "model_manifest.json",
    "poster_setting_recovery.md",
    "run_manifest.json",
    "runtime_audit.csv",
    "sample_convergence.csv",
    "field_dependence.csv",
    "field_metrics.csv",
    "overhauser_curves.csv",
    "overhauser_metrics.csv",
    "overhauser_dynamics.csv",
    "entanglement_scaling.csv",
    "entanglement_fit_parameters.csv",
    "two_electron_entanglement.csv",
    "n10_control_curves.csv",
    "n10_control_metrics.csv",
    "n10_control_dynamics.csv",
    "summary.md",
)

REQUIRED_FIGURES = (
    "figure2_field_dependence",
    "figure2b_contrast_vs_field",
    "figure2c_field_angle_heatmap",
    "figure3_overhauser_comparison",
    "figure3b_overhauser_error_vs_field",
    "figure3c_overhauser_dynamics",
    "figure4_entanglement_scaling",
    "figure4b_two_electron_entanglement",
    "figure4c_initial_state_check",
    "figure5_n10_control",
    "figure5b_n10_slope",
    "figure5c_n10_dynamics",
)


def completion_snapshot(
    output: Path,
    *,
    samples: int = PRODUCTION_MODEL.samples,
    overhauser_samples: int = 8_388_608,
    overhauser_replicates: int = 8,
    atomwise_samples: int = 4096,
    seed: int = MASTER_SEED,
) -> dict[str, object]:
    tasks = planned_tasks(
        "production",
        samples,
        overhauser_samples,
        seed,
        (2, 3, 4, 5),
        overhauser_replicates,
        atomwise_samples,
    )
    unique_tasks = {(task.kind, task.key): task for task in tasks}
    expected_by_kind = Counter(task.kind for task in unique_tasks.values())
    completed_by_kind = Counter(
        task.kind
        for task in unique_tasks.values()
        if cell_path(output, task).exists()
    )
    missing_by_kind = {
        kind: expected_by_kind[kind] - completed_by_kind[kind]
        for kind in sorted(expected_by_kind)
    }

    coarse_quantum_complete = completed_by_kind["quantum"] == expected_by_kind["quantum"]
    refinement_expected = refinement_completed = 0
    refinement_plannable = False
    if coarse_quantum_complete:
        refinement_tasks = planned_refinement_tasks(output, samples, seed)
        refinement_expected = len(refinement_tasks)
        refinement_completed = sum(
            cell_path(output, task).exists() for task in refinement_tasks
        )
        refinement_plannable = True

    artifact_status = {
        name: (output / name).exists() for name in REQUIRED_TABLES
    }
    for stem in REQUIRED_FIGURES:
        for extension in ("png", "pdf"):
            relative = f"figures/{stem}.{extension}"
            artifact_status[relative] = (output / relative).exists()

    model_checks = {
        "manifest_present": False,
        "configuration_hash_matches": False,
        "production_samples_match": False,
        "factorized_evolution_false": False,
        "electron_pair_coupling_retained": False,
        "quadrupole_scale_zero": False,
    }
    manifest_path = output / "model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        representation = manifest.get("hamiltonian_representation", {})
        model_checks.update(
            {
                "manifest_present": True,
                "configuration_hash_matches": manifest.get("configuration_hash")
                == PRODUCTION_MODEL.configuration_hash,
                "production_samples_match": manifest.get("sampler", {}).get(
                    "production_samples"
                )
                == samples,
                "factorized_evolution_false": representation.get(
                    "factorized_evolution"
                )
                is False,
                "electron_pair_coupling_retained": representation.get(
                    "electron_pair_coupling_retained"
                )
                is True,
                "quadrupole_scale_zero": manifest.get("quadrupole_scale") == 0.0,
            }
        )

    production_complete = all(value == 0 for value in missing_by_kind.values())
    refinement_complete = (
        refinement_plannable and refinement_completed == refinement_expected
    )
    artifacts_complete = all(artifact_status.values())
    model_valid = all(model_checks.values())
    return {
        "schema_version": 1,
        "output_directory": str(output.resolve()),
        "samples": samples,
        "overhauser_samples": overhauser_samples,
        "expected_production_cells": len(unique_tasks),
        "completed_production_cells": sum(completed_by_kind.values()),
        "expected_by_kind": dict(sorted(expected_by_kind.items())),
        "completed_by_kind": {
            kind: completed_by_kind[kind] for kind in sorted(expected_by_kind)
        },
        "missing_by_kind": missing_by_kind,
        "coarse_quantum_complete": coarse_quantum_complete,
        "refinement_plannable": refinement_plannable,
        "refinement_expected_cells": refinement_expected,
        "refinement_completed_cells": refinement_completed,
        "production_complete": production_complete,
        "refinement_complete": refinement_complete,
        "model_checks": model_checks,
        "artifact_status": artifact_status,
        "artifacts_complete": artifacts_complete,
        "complete": production_complete
        and refinement_complete
        and model_valid
        and artifacts_complete,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=PRODUCTION_MODEL.samples)
    parser.add_argument("--overhauser-samples", type=int, default=8_388_608)
    parser.add_argument("--overhauser-replicates", type=int, default=8)
    parser.add_argument("--atomwise-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = completion_snapshot(
        args.output_dir,
        samples=args.samples,
        overhauser_samples=args.overhauser_samples,
        overhauser_replicates=args.overhauser_replicates,
        atomwise_samples=args.atomwise_samples,
        seed=args.seed,
    )
    atomic_json(args.output_dir / "completion_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_complete and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
