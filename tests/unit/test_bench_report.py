# pyright: reportPrivateUsage=false

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from inundation.bench.report import (
    SUITES,
    SuiteResult,
    _description_details,
    render_html,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_render_html_includes_collapsed_escaped_suite_descriptions() -> None:
    results = [
        SuiteResult(
            spec=spec,
            exit_code=0,
            wall_s=0.0,
            log_text="",
            summary={},
        )
        for spec in SUITES
    ]

    rendered = render_html(None, results)

    assert rendered.count("<summary>Test description</summary>") == len(SUITES)
    assert "<details open>" not in rendered
    for spec in SUITES:
        assert html.escape(spec.description).replace("\n", "<br/>") in rendered


def test_description_details_escapes_html_significant_text() -> None:
    rendered = _description_details("Test description", "<script>alert('&')</script>")

    assert rendered == (
        "<details><summary>Test description</summary>"
        "<p class='description'>&lt;script&gt;alert(&#x27;&amp;&#x27;)&lt;/script&gt;</p></details>"
    )


def test_render_html_adds_column_hover_descriptions() -> None:
    result = SuiteResult(
        spec=SUITES[0],
        exit_code=0,
        wall_s=0.0,
        log_text="",
        summary={
            "gates": {"volume": 1e-5},
            "correctness": [
                {
                    "case": "synthetic",
                    "backend": "cpu",
                    "front_rel_err": 0.1,
                    "isotropy_spread": 0.02,
                    "ponded_storage_frac": 0.5,
                    "point2_depth_m": 0.3,
                    "max_arrival_err_rel": 0.04,
                    "fail_reasons": [],
                    "<future&field>": "value",
                    "passed": True,
                }
            ],
            "performance": [
                {
                    "case": "synthetic",
                    "backend": "cpu",
                    "median_wall_s": 1.2,
                    "repeats": 2,
                    "passed": True,
                }
            ],
        },
    )

    rendered = render_html(None, [result])

    assert 'title="Benchmark suite name."' in rendered
    assert 'title="Named acceptance gate evaluated by the benchmark."' in rendered
    assert 'title="Benchmark scenario or sub-case."' in rendered
    assert 'title="Whether this reported check passed."' in rendered
    assert (
        'title="Relative error in detected wet/bore-front position against the '
        'analytical/reference front."' in rendered
    )
    assert 'title="Relative spread of radial-front radii across angular sectors."' in rendered
    assert 'title="Fraction of injected volume retained in pond storage."' in rendered
    assert (
        'title="Second depression depth whose ponding is the '
        'momentum-over-crest signature, in metres."' in rendered
    )
    assert (
        'title="Maximum relative gauge-arrival-time error against the '
        'axisymmetric radial reference."' in rendered
    )
    assert 'title="Median elapsed wall-clock time across timed repeats, in seconds."' in rendered
    assert 'title="Reasons the benchmark gates failed, if any."' in rendered
    assert 'title="Undocumented benchmark field: &lt;future&amp;field&gt;."' in rendered
    assert "&lt;future&amp;field&gt;" in rendered


def test_render_html_inlines_graphs_before_animations(tmp_path: Path) -> None:
    from inundation.bench.report import _suite_section

    graph = tmp_path / "bad<&graph.png"
    gif = tmp_path / "animation.gif"
    graph.write_bytes(b"png")
    gif.write_bytes(b"gif")
    result = SuiteResult(
        spec=SUITES[0],
        exit_code=0,
        wall_s=0.0,
        log_text="",
        summary={},
        graph_paths=[graph],
        gif_paths=[gif],
    )
    rendered = _suite_section(result)
    assert "Graphs" in rendered and "Animations" in rendered
    assert rendered.index("Graphs") < rendered.index("Animations")
    assert "data:image/png;base64" in rendered
    assert "bad&lt;&amp;graph.png" in rendered


def test_the_wetting_and_drying_case_is_a_gate() -> None:
    assert "parabolic_bowl" in {spec.name for spec in SUITES}
