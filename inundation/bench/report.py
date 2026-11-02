"""Run the full benchmark suite (+ unit tests) and emit a standalone HTML report.

Each gated harness (`dambreak`, `radial_dambreak`, `floodplain_depressions`,
`momentum_obstruction`, `flood_propagation`, `open_boundary`) is run as a subprocess through its
own CLI with GIF rendering enabled, so the metrics and animations are exactly
the ones the harness itself produces — no reimplementation. The per-suite
``summary.json`` files, console logs and GIF animations are then folded into a
single self-contained ``report.html`` (GIFs are base64-inlined, so the file can
be shared as-is).

The report is written even when suites fail; the process exit code reflects the
overall verdict (0 = everything passed).

Run (macOS host, MoltenVK — needs an unsandboxed shell)::

    .venv/bin/python -m inundation.bench.report

or through the dev container::

    just benchmark-report
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

DEFAULT_OUTPUT_ROOT = ".tmp/bench-report"


@dataclass(frozen=True)
class SuiteSpec:
    """One benchmark harness and the flags needed to run it with animations."""

    name: str
    module: str
    title: str
    description: str
    gif_args: tuple[str, ...]


SUITES: tuple[SuiteSpec, ...] = (
    SuiteSpec(
        name="dambreak",
        module="inundation.bench.dambreak",
        title="1D dam-break (Stoker/Ritter analytical)",
        description=(
            "Purpose: 1D analytical dam-break regression against the Stoker and Ritter solutions.\n"
            "Setup: 1000 m × 10 m flat channel with the dam at x=500 m; upstream depth is 10 m, "
            "and the Stoker downstream depth is 1 m while the Ritter case is dry-bed.\n"
            "Forcing and runtime: instantaneous dam removal, frictionless/minimal Manning floor, "
            "20 s run.\n"
            "Validation: depth-profile, wet-front, volume-conservation, and determinism gates."
        ),
        gif_args=("--gif",),
    ),
    SuiteSpec(
        name="radial_dambreak",
        module="inundation.bench.radial_dambreak",
        title="Radial dam-break (axisymmetric FV reference)",
        description=(
            "Purpose: circular 2D dam-break regression and Cartesian-grid isotropy check.\n"
            "Setup: 40 m × 40 m flat square with a centered radius-2.5 m water column; inner "
            "depth is 2.5 m and outer depth is 0.5 m.\n"
            "Forcing and runtime: instantaneous collapse over minimal friction, 1.5 s run.\n"
            "Reference and validation: compared with a fine radial reference using depth-error, "
            "front-radius, isotropy, volume-conservation, and determinism gates."
        ),
        gif_args=("--gif", "--gif-2d"),
    ),
    SuiteSpec(
        name="floodplain_depressions",
        module="inundation.bench.floodplain_depressions",
        title="EA Test 2 — floodplain depressions",
        description=(
            "Purpose: EA Test 2 wetting/drying and basin-filling case.\n"
            "Setup: 2000 m × 2000 m DEM domain at 20 m resolution, initially dry, with Manning "
            "n=0.03.\n"
            "Forcing and runtime: western-edge inflow represented by near-boundary volume sources "
            "following the published hydrograph, 48 h run.\n"
            "Validation: volume conservation, pond storage and sill behavior, far-column dryness, "
            "and determinism gates."
        ),
        gif_args=("--gif",),
    ),
    SuiteSpec(
        name="momentum_obstruction",
        module="inundation.bench.momentum_obstruction",
        title="EA Test 3 — momentum over an obstruction",
        description=(
            "Purpose: EA Test 3 momentum-over-obstruction case.\n"
            "Setup: 300 m × 100 m DEM domain at 2 m resolution, initially dry, Manning n=0.01, "
            "with two depressions separated by a crest.\n"
            "Forcing and runtime: the published hydrograph is represented by near-boundary volume "
            "sources, 900 s run.\n"
            "Validation: volume, pond-level, control, and determinism gates. The first pond and "
            "crest isolation are controls; water retained in the second depression is the momentum "
            "signature."
        ),
        gif_args=("--gif",),
    ),
    SuiteSpec(
        name="flood_propagation",
        module="inundation.bench.flood_propagation",
        title="EA Test 4 — flood-front propagation",
        description=(
            "Purpose: EA Test 4 flood-front-speed case.\n"
            "Setup: 1000 m × 2000 m flat dry domain at 5 m resolution with Manning n=0.05.\n"
            "Forcing and runtime: western-wall line inflow represented by volume sources following "
            "the published hydrograph, 5 h run.\n"
            "Reference and validation: at 2 h, compare with an axisymmetric radial reference for "
            "arrival time, depth, speed, isotropy, volume conservation, and determinism."
        ),
        gif_args=("--gif",),
    ),
    SuiteSpec(
        name="parabolic_bowl",
        module="inundation.bench.parabolic_bowl",
        title="Thacker parabolic bowl (exact wetting and drying)",
        description=(
            "Purpose: repeated wetting and drying against an exact solution.\n"
            "Setup: frictionless paraboloid, 4 m × 4 m at 200 × 200 cells, tilted planar "
            "surface, so the shoreline moves across cells every period.\n"
            "Forcing and runtime: none, three periods.\n"
            "Validation: depth error, thin-film speed, clamp mass, volume, and determinism gates."
        ),
        gif_args=(),
    ),
    SuiteSpec(
        name="basin_exchange",
        module="inundation.bench.basin_exchange",
        title="Basin exchange (water both ways between meshes)",
        description=(
            "Purpose: basins on separate meshes against one mesh over the same area.\n"
            "Setup: straight channels cut in two, and a plain cut in three. A lake across a "
            "border, a dam break through it, M1 backwater, flow reversing, a bed step at the "
            "border and a confluence.\n"
            "Forcing and runtime: discharge inflow, stage and point sources, 2 s windows.\n"
            "Validation: depth against one mesh, level and speed at rest, system balance, "
            "clamp mass, and determinism."
        ),
        gif_args=(),
    ),
    SuiteSpec(
        name="open_boundary",
        module="inundation.bench.open_boundary",
        title="Open boundaries (analytical cases and basin exchange)",
        description=(
            "Purpose: water leaving and entering a mesh at named stretches.\n"
            "Setup: straight channels. Uniform flow, Ritter and Stoker dam breaks leaving a "
            "cut, a lake held by its own level, M1 and M2 backwater, and one channel run "
            "whole and in two basins.\n"
            "Forcing and runtime: discharge inflow, normal depth, stage and free outflow.\n"
            "Validation: analytic depth, measured volume balance, clamp mass, and the volume "
            "passed between basins."
        ),
        gif_args=(),
    ),
)


@dataclass
class SuiteResult:
    """Everything the report needs about one completed suite run."""

    spec: SuiteSpec
    exit_code: int
    wall_s: float
    log_text: str
    summary: dict[str, object] | None
    gif_paths: list[Path] = field(default_factory=lambda: list[Path]())
    graph_paths: list[Path] = field(default_factory=lambda: list[Path]())

    @property
    def status(self) -> str:
        if self.exit_code == 0 and self.summary is not None:
            return "PASS"
        if self.exit_code == 1 and self.summary is not None:
            return "FAIL"
        return "ERROR"


@dataclass
class PytestResult:
    """Outcome of the unit-test run."""

    exit_code: int
    wall_s: float
    log_text: str


# ── Suite execution ──────────────────────────────────────────────────────────


def _run_suite(spec: SuiteSpec, output_root: Path, extra_args: list[str]) -> SuiteResult:
    output_dir = output_root / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        spec.module,
        "--output-dir",
        str(output_dir),
        *spec.gif_args,
        *extra_args,
    ]
    print(f"[{spec.name}] running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        cmd, check=False, capture_output=True, text=True
    )
    wall_s = time.perf_counter() - t0
    log_text = completed.stdout + (("\n" + completed.stderr) if completed.stderr else "")

    summary: dict[str, object] | None = None
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        parsed: object = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            summary = cast("dict[str, object]", parsed)

    result = SuiteResult(
        spec=spec,
        exit_code=completed.returncode,
        wall_s=wall_s,
        log_text=log_text,
        summary=summary,
        gif_paths=sorted(output_dir.glob("*.gif")),
        graph_paths=sorted(output_dir.glob("*.png")),
    )
    print(
        f"[{spec.name}] {result.status} in {wall_s:.1f} s "
        f"(exit {completed.returncode}, {len(result.gif_paths)} gif(s))",
        flush=True,
    )
    return result


def _run_pytest(repo_root: Path) -> PytestResult:
    cmd = [sys.executable, "-m", "pytest", "tests", "-q", "--color=no"]
    print(f"[unit-tests] running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        cmd, check=False, capture_output=True, text=True, cwd=repo_root
    )
    wall_s = time.perf_counter() - t0
    log_text = completed.stdout + (("\n" + completed.stderr) if completed.stderr else "")
    verdict = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"[unit-tests] {verdict} in {wall_s:.1f} s (exit {completed.returncode})", flush=True)
    return PytestResult(exit_code=completed.returncode, wall_s=wall_s, log_text=log_text)


# ── HTML rendering ───────────────────────────────────────────────────────────

_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
       background: #f6f7f9; color: #1f2328; }
main { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.2rem; margin-top: 2.2rem; }
.meta { color: #57606a; font-size: 0.85rem; }
.badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 1rem;
         font-weight: 600; font-size: 0.8rem; color: #fff; vertical-align: middle; }
.badge.pass { background: #1a7f37; } .badge.fail { background: #cf222e; }
.badge.error { background: #bf8700; } .badge.skip { background: #57606a; }
table { border-collapse: collapse; margin: 0.6rem 0 1rem; font-size: 0.85rem;
        background: #fff; }
th, td { border: 1px solid #d0d7de; padding: 0.3rem 0.6rem; text-align: left;
         white-space: nowrap; }
th { background: #eaeef2; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.failed td { background: #ffebe9; }
details { margin: 0.6rem 0; }
details > summary { cursor: pointer; color: #57606a; font-size: 0.85rem; }
.description { margin: 0.5rem 0 1rem; white-space: normal; line-height: 1.5; }
pre { background: #24292f; color: #d0d7de; padding: 0.8rem; border-radius: 6px;
      overflow-x: auto; font-size: 0.75rem; line-height: 1.4; }
.gifs { display: flex; flex-wrap: wrap; gap: 1rem; }
figure { margin: 0; }
figure img { max-width: 520px; width: 100%; border: 1px solid #d0d7de;
             border-radius: 6px; }
figcaption { font-size: 0.8rem; color: #57606a; margin-top: 0.2rem; }
"""


def _badge(status: str) -> str:
    return f'<span class="badge {status.lower()}">{html.escape(status)}</span>'


def _fmt_value(value: object) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "&#10003;" if value else "&#10007;"
    if isinstance(value, float):
        return html.escape(f"{value:.6g}")
    if isinstance(value, list):
        items = cast("list[object]", value)
        return html.escape(", ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in items))
    return html.escape(str(value))


_COLUMN_DESCRIPTIONS = {
    "suite": "Benchmark suite name.",
    "status": "Overall suite verdict.",
    "wall s": "Elapsed wall-clock time for the suite, in seconds.",
    "cases passed": "Correctness checks passed out of total checks.",
    "gifs": "Number of generated animations.",
    "gate": "Named acceptance gate evaluated by the benchmark.",
    "threshold": "Acceptance limit or target for the gate.",
    "passed": "Whether this reported check passed.",
    "case": "Benchmark scenario or sub-case.",
    "backend": "Solver backend used for the run.",
    "repeats": "Number of timed solver repeats used for the performance summary.",
    "median_wall_s": "Median elapsed wall-clock time across timed repeats, in seconds.",
    "steps_total": "Total solver time steps completed by the timed run.",
    "steps_per_s": "Solver time steps completed per second of wall-clock time.",
    "cell_steps_per_s": "Cell updates completed per second of wall-clock time.",
    "deterministic": "Whether repeated runs produced identical benchmark results.",
    "determinism_detail": "Explanation of the repeated-run determinism comparison.",
    "l1_rel": "Relative L1 error against the analytical or reference solution.",
    "l2_rel": "Relative L2 error against the analytical or reference solution.",
    "front_rel_err": "Relative error in detected wet/bore-front position against the analytical/reference front.",
    "isotropy_spread": "Relative spread of radial-front radii across angular sectors.",
    "volume_drift_rel": "Relative change in conserved water volume over the run.",
    "min_depth_m": "Minimum simulated water depth, in metres.",
    "h_final_finite": "Whether all final water-depth values are finite.",
    "fail_reasons": "Reasons the benchmark gates failed, if any.",
    "ponded_count": "Number of depressions retaining ponded water above their sills.",
    "max_above_sill_m": "Maximum water depth above a depression sill, in metres.",
    "full_basin_count": "Number of depressions filled to their measured basin capacity.",
    "full_basin_capacity_rel": "Relative capacity error for depressions classified as full.",
    "ponded_storage_frac": "Fraction of injected volume retained in pond storage.",
    "far_column_max_depth_m": "Maximum water depth in the far column away from the inflow, in metres.",
    "point_depths_m": "Water depths at the configured depression probe points, in metres.",
    "sill_depths_m": "Water depths at the configured depression sill points, in metres.",
    "crest_z_m": "Obstruction crest elevation, in metres.",
    "point1_depth_m": "Water depth at the first obstruction probe point, in metres.",
    "point1_wse_m": "Water-surface elevation at the first obstruction probe point, in metres.",
    "crest_depth_m": "Water depth at the obstruction crest, in metres.",
    "point2_depth_m": "Second depression depth whose ponding is the momentum-over-crest signature, in metres.",
    "point2_wse_m": "Water-surface elevation at the second obstruction probe point, in metres.",
    "control_point2_depth_m": "Water depth at the dry control point corresponding to the second probe, in metres.",
    "left_ponded": "Whether the left obstruction-side depression is ponded.",
    "ponds_disconnected": "Whether the two obstruction-side ponds remain hydraulically disconnected.",
    "point2_risen": "Whether the second obstruction probe shows the expected risen water level.",
    "control_dry": "Whether the obstruction control point remains dry.",
    "volume_final_m3": "Final simulated water volume, in cubic metres.",
    "volume_injected_m3": "Water volume injected by the propagation source, in cubic metres.",
    "gauge_radii_m": "Radial distances of the propagation gauges from the source, in metres.",
    "arrival_s": "Simulated arrival times at the propagation gauges, in seconds.",
    "arrival_ref_s": "Reference radial arrival times at the propagation gauges, in seconds.",
    "max_arrival_err_rel": "Maximum relative gauge-arrival-time error against the axisymmetric radial reference.",
    "probe_depth_m": "Simulated water depths at propagation probes, in metres.",
    "probe_depth_ref_m": "Reference water depths at propagation probes, in metres.",
    "depth_l1_rel": "Relative L1 error of propagation-probe depths against the reference.",
    "probe_speed_ms": "Simulated propagation speeds at probes, in metres per second.",
    "probe_speed_ref_ms": "Reference propagation speeds at probes, in metres per second.",
    "speed_l1_rel": "Relative L1 error of propagation-probe speeds against the reference.",
    "isotropy_excess_m": "Excess radial-front spread beyond the reference isotropy, in metres.",
}


def _column_header(name: str) -> str:
    description = _COLUMN_DESCRIPTIONS.get(name, f"Undocumented benchmark field: {name}.")
    return f'<th title="{html.escape(description, quote=True)}">{html.escape(name)}</th>'


def _rows_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p class='meta'>no entries</p>"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    head = "".join(_column_header(c) for c in columns)
    body_rows: list[str] = []
    for row in rows:
        failed = row.get("passed") is False
        cells = "".join(
            f"<td class='{'num' if isinstance(row.get(c), int | float) else ''}'>"
            f"{_fmt_value(row.get(c))}</td>"
            for c in columns
        )
        body_rows.append(f"<tr class='{'failed' if failed else ''}'>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _mapping_table(mapping: dict[str, object]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{_fmt_value(v)}</td></tr>"
        for k, v in mapping.items()
    )
    return f"<table><thead><tr>{_column_header('gate')}{_column_header('threshold')}</tr></thead><tbody>{body}</tbody></table>"


def _as_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    entries = cast("list[object]", value)
    return [cast("dict[str, object]", e) for e in entries if isinstance(e, dict)]


def _log_details(title: str, log_text: str) -> str:
    return (
        f"<details><summary>{html.escape(title)}</summary>"
        f"<pre>{html.escape(log_text.strip() or '(no output)')}</pre></details>"
    )


def _gif_figure(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f"<figure><img loading='lazy' src='data:image/gif;base64,{data}' "
        f"alt='{html.escape(path.name)}'/>"
        f"<figcaption>{html.escape(path.name)} ({path.stat().st_size / 1e6:.1f} MB)</figcaption>"
        f"</figure>"
    )


def _png_figure(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f"<figure><img loading='lazy' src='data:image/png;base64,{data}' "
        f"alt='{html.escape(path.name)}'/>"
        f"<figcaption>{html.escape(path.name)} ({path.stat().st_size / 1e3:.0f} kB)</figcaption>"
        f"</figure>"
    )


def _description_details(title: str, description: str) -> str:
    escaped = html.escape(description).replace("\n", "<br/>")
    return f"<details><summary>{html.escape(title)}</summary><p class='description'>{escaped}</p></details>"


def _suite_section(result: SuiteResult) -> str:
    parts: list[str] = [
        f"<h2 id='{html.escape(result.spec.name)}'>"
        f"{html.escape(result.spec.title)} {_badge(result.status)}</h2>",
        f"<p class='meta'>module <code>{html.escape(result.spec.module)}</code> · "
        f"wall {result.wall_s:.1f} s · exit {result.exit_code}</p>",
        _description_details("Test description", result.spec.description),
    ]
    if result.summary is not None:
        gates = result.summary.get("gates")
        if isinstance(gates, dict):
            parts.append("<h3>Gates</h3>")
            parts.append(_mapping_table(cast("dict[str, object]", gates)))
        correctness = _as_rows(result.summary.get("correctness"))
        if correctness:
            parts.append("<h3>Correctness</h3>")
            parts.append(_rows_table(correctness))
        performance = _as_rows(result.summary.get("performance"))
        if performance:
            parts.append("<h3>Performance</h3>")
            parts.append(_rows_table(performance))
    else:
        parts.append("<p class='meta'>no summary.json produced — see the log below</p>")
    if result.graph_paths:
        parts.append("<h3>Graphs</h3><div class='graphs'>")
        parts.extend(_png_figure(p) for p in result.graph_paths)
        parts.append("</div>")
    if result.gif_paths:
        parts.append("<h3>Animations</h3><div class='gifs'>")
        parts.extend(_gif_figure(p) for p in result.gif_paths)
        parts.append("</div>")
    parts.append(_log_details("console log", result.log_text))
    return "\n".join(parts)


def _overview_table(pytest_result: PytestResult | None, results: list[SuiteResult]) -> str:
    rows: list[str] = []
    if pytest_result is not None:
        status = "PASS" if pytest_result.exit_code == 0 else "FAIL"
        rows.append(
            f"<tr><td><a href='#unit-tests'>unit tests (pytest)</a></td>"
            f"<td>{_badge(status)}</td><td class='num'>{pytest_result.wall_s:.1f}</td>"
            f"<td class='num'>&mdash;</td><td class='num'>&mdash;</td></tr>"
        )
    for r in results:
        correctness = _as_rows(r.summary.get("correctness")) if r.summary else []
        passed = sum(1 for row in correctness if row.get("passed") is True)
        cases = f"{passed}/{len(correctness)}" if correctness else "&mdash;"
        rows.append(
            f"<tr><td><a href='#{html.escape(r.spec.name)}'>{html.escape(r.spec.title)}</a></td>"
            f"<td>{_badge(r.status)}</td><td class='num'>{r.wall_s:.1f}</td>"
            f"<td class='num'>{cases}</td><td class='num'>{len(r.gif_paths)}</td></tr>"
        )
    headers = ("suite", "status", "wall s", "cases passed", "gifs")
    return (
        f"<table><thead><tr>{''.join(_column_header(c) for c in headers)}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(pytest_result: PytestResult | None, results: list[SuiteResult]) -> str:
    """Render the full standalone report (GIFs inlined as data URIs)."""
    overall_pass = all(r.status == "PASS" for r in results) and (
        pytest_result is None or pytest_result.exit_code == 0
    )
    overall = _badge("PASS" if overall_pass else "FAIL")
    generated = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        f"<h1>Inundation — benchmark report {overall}</h1>",
        f"<p class='meta'>generated {generated} · {html.escape(platform.platform())} · "
        f"python {html.escape(platform.python_version())}</p>",
        _overview_table(pytest_result, results),
    ]
    if pytest_result is not None:
        status = "PASS" if pytest_result.exit_code == 0 else "FAIL"
        sections.append(f"<h2 id='unit-tests'>Unit tests (pytest) {_badge(status)}</h2>")
        sections.append(
            f"<p class='meta'>wall {pytest_result.wall_s:.1f} s · exit {pytest_result.exit_code}</p>"
        )
        sections.append(_log_details("pytest output", pytest_result.log_text))
    sections.extend(_suite_section(r) for r in results)
    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>"
        "<title>Inundation — benchmark report</title>"
        f"<style>{_CSS}</style></head><body><main>{body}</main></body></html>"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all gated benchmarks (+ unit tests) and emit an HTML report"
    )
    parser.add_argument(
        "--suites",
        default=",".join(s.name for s in SUITES),
        help=f"CSV of {tuple(s.name for s in SUITES)}",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repeats", type=int, default=2, help=">=2 enables determinism checks")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--skip-unit-tests", action="store_true", help="skip the pytest run, benchmarks only"
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=None,
        help="CFL number passed to every suite (default: each suite's)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    wanted = [s.strip() for s in args.suites.split(",") if s.strip()]
    by_name = {s.name: s for s in SUITES}
    unknown = [s for s in wanted if s not in by_name]
    if unknown:
        print(f"unknown suite(s) {unknown}; choose from {tuple(by_name)}")
        return 2

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    pytest_result = None if args.skip_unit_tests else _run_pytest(repo_root)

    extra_args = ["--repeats", str(args.repeats), "--warmup", str(args.warmup)]
    if args.cfl is not None:
        extra_args += ["--cfl", str(args.cfl)]
    results = [_run_suite(by_name[name], output_root, extra_args) for name in wanted]

    report_path = output_root / "report.html"
    report_path.write_text(render_html(pytest_result, results), encoding="utf-8")
    print(f"\n[report] {report_path} ({report_path.stat().st_size / 1e6:.1f} MB)")

    for r in results:
        print(f"  {r.spec.name:24} {r.status}")
    if pytest_result is not None:
        print(f"  {'unit-tests':24} {'PASS' if pytest_result.exit_code == 0 else 'FAIL'}")

    overall_ok = all(r.status == "PASS" for r in results) and (
        pytest_result is None or pytest_result.exit_code == 0
    )
    print(f"RESULT: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
