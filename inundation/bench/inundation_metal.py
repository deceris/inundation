"""Benchmark inundation solver variants on Metal with safety invariants."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from inundation.bench.common import (
    RunFingerprint,
    all_invariants_ok,
    assert_deterministic,
    build_fingerprint,
    build_pipeline_like_phases,
    compare_fingerprints_exact,
    save_depth_png,
)
from inundation.bench.common import (
    median_wall as _median_wall,
)
from inundation.workflow import (
    SWEWorkflow,
    WorkflowConfig,
)

DEFAULT_MESH_PATH = (
    "api/seed/tool/.state/tasks/flood_mesh/downloads/powiat-klodzki-mesh-280k.parquet"
)
DEFAULT_OUTPUT_ROOT = ".tmp/inundation-metal-bench"
AVAILABLE_SOLVER_IMPLS = (
    "batched_submit",
    "async_sync_window",
    "gpu_resident_batch",
)
DEFAULT_SOLVER_IMPLS = ("batched_submit", "gpu_resident_batch")
ComparedSolver = Literal["batched_submit", "async_sync_window", "gpu_resident_batch"]


def _run_once(
    mesh_path: Path,
    solver_impl: Literal["baseline", "batched_submit", "async_sync_window", "gpu_resident_batch"],
    run_index: int,
    solver_progress: bool,
    phase_duration_s: float,
    output_interval_s: float,
    dt_max: float,
    cfl_interval: int,
    plot_path: Path | None = None,
) -> RunFingerprint:
    config = WorkflowConfig(
        mesh_source=str(mesh_path),
        manning_n=0.035,
        output_interval_s=output_interval_s,
        dt_max=dt_max,
        cfl_interval=cfl_interval,
        progress=solver_progress,
        solver_impl=solver_impl,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    result = workflow.run(build_pipeline_like_phases(workflow, phase_duration_s))

    if plot_path is not None:
        snap_times_for_plot = np.asarray(result.snap_times, dtype=np.float64)
        h_final_for_plot = np.asarray(result.h_final, dtype=np.float32)
        save_depth_png(workflow, h_final_for_plot, float(snap_times_for_plot[-1]), plot_path)

    return build_fingerprint(
        run_index=run_index,
        solver_impl=solver_impl,
        result=result,
        expected_final_time_s=phase_duration_s * 2.0,
    )


def _run_group(
    mesh_path: Path,
    solver_impl: Literal["baseline", "batched_submit", "async_sync_window", "gpu_resident_batch"],
    repeats: int,
    warmup: int,
    output_dir: Path,
    solver_progress: bool,
    phase_duration_s: float,
    output_interval_s: float,
    dt_max: float,
    cfl_interval: int,
    plot_dir: Path | None = None,
) -> tuple[list[RunFingerprint], float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = warmup + repeats
    completed = 0
    wall_samples: list[float] = []

    for warmup_index in range(warmup):
        sys.stdout.write(
            f"[{solver_impl}] warmup {warmup_index + 1}/{warmup} "
            f"(overall {completed + 1}/{total_runs}) start\n"
        )
        sys.stdout.flush()
        warmup_fp = _run_once(
            mesh_path,
            solver_impl,
            run_index=-1 - warmup_index,
            solver_progress=solver_progress,
            phase_duration_s=phase_duration_s,
            output_interval_s=output_interval_s,
            dt_max=dt_max,
            cfl_interval=cfl_interval,
        )
        wall_samples.append(warmup_fp.wall_seconds)
        completed += 1
        avg_wall = statistics.fmean(wall_samples)
        eta = max(avg_wall * (total_runs - completed), 0.0)
        sys.stdout.write(
            f"[{solver_impl}] warmup {warmup_index + 1}/{warmup} done "
            f"({warmup_fp.wall_seconds:.2f}s, eta~{eta:.1f}s, "
            f"inv={warmup_fp.invariant_reason})\n"
        )
        sys.stdout.flush()

    runs: list[RunFingerprint] = []
    for run_index in range(repeats):
        sys.stdout.write(
            f"[{solver_impl}] measured {run_index + 1}/{repeats} "
            f"(overall {completed + 1}/{total_runs}) start\n"
        )
        sys.stdout.flush()
        fingerprint = _run_once(
            mesh_path,
            solver_impl,
            run_index=run_index,
            solver_progress=solver_progress,
            phase_duration_s=phase_duration_s,
            output_interval_s=output_interval_s,
            dt_max=dt_max,
            cfl_interval=cfl_interval,
            plot_path=(plot_dir / f"water_depth_{solver_impl}_run{run_index}.png")
            if plot_dir is not None
            else None,
        )
        runs.append(fingerprint)
        wall_samples.append(fingerprint.wall_seconds)
        completed += 1
        avg_wall = statistics.fmean(wall_samples)
        eta = max(avg_wall * (total_runs - completed), 0.0)
        sys.stdout.write(
            f"[{solver_impl}] measured {run_index + 1}/{repeats} done "
            f"({fingerprint.wall_seconds:.2f}s, eta~{eta:.1f}s, "
            f"inv={fingerprint.invariant_reason})\n"
        )
        sys.stdout.flush()
        (output_dir / f"run_{run_index}.json").write_text(
            json.dumps(asdict(fingerprint), indent=2),
            encoding="utf-8",
        )

    median_wall_value = _median_wall(runs)
    return runs, median_wall_value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default=DEFAULT_MESH_PATH)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--solver-progress", action="store_true")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--require-bitwise-parity", action="store_true")
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=AVAILABLE_SOLVER_IMPLS,
        default=list(DEFAULT_SOLVER_IMPLS),
    )
    parser.add_argument("--phase-duration-s", type=float, default=300.0)
    parser.add_argument("--output-interval-s", type=float, default=60.0)
    parser.add_argument("--dt-max", type=float, default=0.5)
    parser.add_argument("--cfl-interval", type=int, default=10)
    parser.add_argument(
        "--plot-dir",
        default=None,
        help=(
            "If set, save a water-depth PNG per solver impl (first measured run) "
            'to this directory. Requires matplotlib: uv pip install -e ".[viz]"'
        ),
    )
    return parser


def _validate_runtime_deps() -> None:
    """Fail fast with actionable message when GPU benchmark deps are missing."""
    missing: list[str] = []
    for module_name in ("kp", "hilbertcurve"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing benchmark dependencies: {missing_csv}. "
            "Use GPU worker environment (.venv-gpu). Run: just gpu-worker::setup"
        )


def main() -> int:
    """Run benchmark and enforce parity + invariants under pipeline-like settings."""
    _validate_runtime_deps()
    args = _build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.phase_duration_s <= 0.0:
        raise ValueError("--phase-duration-s must be > 0")
    if args.output_interval_s <= 0.0:
        raise ValueError("--output-interval-s must be > 0")
    if args.dt_max <= 0.0:
        raise ValueError("--dt-max must be > 0")
    if args.cfl_interval < 1:
        raise ValueError("--cfl-interval must be >= 1")

    # Resolve paths from the working directory first, then the script directory.
    # Does NOT assume a fixed monorepo directory depth (no parents[N]).
    script_dir = Path(__file__).resolve().parent

    def _resolve_path(raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        cwd_candidate = Path.cwd() / p
        if cwd_candidate.exists():
            return cwd_candidate
        script_candidate = script_dir / p
        if script_candidate.exists():
            return script_candidate
        return cwd_candidate

    mesh_path = _resolve_path(args.mesh)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    plot_dir: Path | None = None
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
        if not plot_dir.is_absolute():
            plot_dir = Path.cwd() / plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

    solver_runs: dict[str, list[RunFingerprint]] = {}
    solver_medians: dict[str, float] = {}
    solver_det: dict[str, tuple[bool | None, str]] = {}
    solver_inv: dict[str, tuple[bool, str]] = {}

    for solver_impl in args.solvers:
        runs, median = _run_group(
            mesh_path,
            solver_impl,
            args.repeats,
            args.warmup,
            output_root / solver_impl,
            args.solver_progress,
            args.phase_duration_s,
            args.output_interval_s,
            args.dt_max,
            args.cfl_interval,
            plot_dir,
        )
        solver_runs[solver_impl] = runs
        solver_medians[solver_impl] = median
        solver_det[solver_impl] = assert_deterministic(runs)
        solver_inv[solver_impl] = all_invariants_ok(runs)

    batched_submit_runs = solver_runs.get("batched_submit")
    batched_submit_median = solver_medians.get("batched_submit")
    parity: dict[str, tuple[bool, str]] = {}
    if batched_submit_runs is not None:
        for solver_impl, runs in solver_runs.items():
            if solver_impl == "batched_submit":
                continue
            parity[f"batched_submit_vs_{solver_impl}"] = compare_fingerprints_exact(
                batched_submit_runs[0], runs[0]
            )

    baseline_det_ok = None
    baseline_det_reason = None
    baseline_inv_ok = None
    baseline_inv_reason = None
    baseline_median = None
    baseline_parity: dict[str, tuple[bool | None, str]] = {}

    if args.include_baseline:
        baseline_runs, baseline_median = _run_group(
            mesh_path,
            "baseline",
            args.repeats,
            args.warmup,
            output_root / "baseline",
            args.solver_progress,
            args.phase_duration_s,
            args.output_interval_s,
            args.dt_max,
            args.cfl_interval,
            plot_dir,
        )
        baseline_det_ok, baseline_det_reason = assert_deterministic(baseline_runs)
        baseline_inv_ok, baseline_inv_reason = all_invariants_ok(baseline_runs)
        for solver_impl, runs in solver_runs.items():
            baseline_parity[solver_impl] = compare_fingerprints_exact(baseline_runs[0], runs[0])

    summary = {
        "mesh": str(mesh_path),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "solver_progress": args.solver_progress,
        "solvers": args.solvers,
        "require_bitwise_parity": args.require_bitwise_parity,
        "scenario": {
            "duration_total_s": args.phase_duration_s * 2.0,
            "phase_count": 2,
            "phase_duration_s": args.phase_duration_s,
            "dt_max": args.dt_max,
            "cfl_interval": args.cfl_interval,
            "output_interval_s": args.output_interval_s,
        },
        "solver_deterministic": {key: value[0] for key, value in sorted(solver_det.items())},
        "solver_deterministic_reason": {key: value[1] for key, value in sorted(solver_det.items())},
        "solver_invariants_ok": {key: value[0] for key, value in sorted(solver_inv.items())},
        "solver_invariants_reason": {key: value[1] for key, value in sorted(solver_inv.items())},
        "batched_submit_parity": {key: value[0] for key, value in sorted(parity.items())},
        "batched_submit_parity_reason": {key: value[1] for key, value in sorted(parity.items())},
        "median_wall_seconds": solver_medians,
        "speedup_vs_batched_submit": {
            key: (
                batched_submit_median / value
                if batched_submit_median is not None and value > 0
                else None
            )
            for key, value in sorted(solver_medians.items())
            if key != "batched_submit"
        },
        "include_baseline": args.include_baseline,
        "baseline_deterministic": baseline_det_ok,
        "baseline_deterministic_reason": baseline_det_reason,
        "baseline_invariants_ok": baseline_inv_ok,
        "baseline_invariants_reason": baseline_inv_reason,
        "baseline_median_wall_seconds": baseline_median,
        "baseline_bitwise_parity": {
            key: value[0] for key, value in sorted(baseline_parity.items())
        },
        "baseline_bitwise_parity_reason": {
            key: value[1] for key, value in sorted(baseline_parity.items())
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(f"Summary written to {summary_path}\n")

    has_fail = any(not value[0] for value in solver_det.values())
    has_fail = has_fail or any(not value[0] for value in solver_inv.values())
    if args.require_bitwise_parity:
        has_fail = has_fail or any(not value[0] for value in parity.values())
    if args.include_baseline:
        has_fail = has_fail or (baseline_det_ok is not True)
        has_fail = has_fail or (baseline_inv_ok is not True)
        if args.require_bitwise_parity:
            has_fail = has_fail or any(value[0] is not True for value in baseline_parity.values())

    if has_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
