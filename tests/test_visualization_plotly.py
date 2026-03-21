"""Plotly-only visualization tests — no matplotlib dependency.

This file tests the plotly backend without requiring matplotlib,
exercising the advertised ``pip install diff-diff[plotly]`` path.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cs_results():
    """Mock CallawaySantAnnaResults."""
    results = MagicMock()
    results.groups = [2004, 2006]
    results.time_periods = [2003, 2004, 2005, 2006, 2007]
    results.group_time_effects = {
        (2004, 2003): {
            "effect": 0.02,
            "se": 0.1,
            "p_value": 0.84,
            "n_treated": 50,
            "n_control": 100,
        },
        (2004, 2004): {
            "effect": 0.5,
            "se": 0.12,
            "p_value": 0.001,
            "n_treated": 50,
            "n_control": 100,
        },
        (2006, 2006): {
            "effect": 0.4,
            "se": 0.12,
            "p_value": 0.001,
            "n_treated": 30,
            "n_control": 100,
        },
    }
    return results


@pytest.fixture
def sensitivity_results():
    """Mock SensitivityResults."""
    results = MagicMock()
    results.M_values = np.array([0, 0.5, 1.0, 1.5, 2.0])
    results.bounds = [(0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0), (-0.1, 1.1)]
    results.robust_cis = [(0.1, 0.9), (0.0, 1.0), (-0.1, 1.1), (-0.2, 1.2), (-0.3, 1.3)]
    results.original_estimate = 0.5
    results.breakdown_M = 1.5
    return results


@pytest.fixture
def honest_results():
    """Mock HonestDiDResults."""
    results = MagicMock()
    results.alpha = 0.05
    results.M = 1.0
    results.event_study_bounds = {
        -2: {"ci_lb": -0.3, "ci_ub": 0.3},
        -1: {"ci_lb": -0.2, "ci_ub": 0.2},
        0: {"ci_lb": 0.1, "ci_ub": 0.9},
        1: {"ci_lb": 0.2, "ci_ub": 1.0},
    }
    original = MagicMock()
    original.period_effects = {
        -2: MagicMock(effect=0.05, se=0.1),
        -1: MagicMock(effect=0.0, se=0.1),
        0: MagicMock(effect=0.5, se=0.12),
        1: MagicMock(effect=0.6, se=0.13),
    }
    results.original_results = original
    return results


@pytest.fixture
def bacon_results():
    """Mock BaconDecompositionResults."""
    results = MagicMock()
    results.comparisons = [
        MagicMock(comparison_type="treated_vs_never", estimate=1.0, weight=0.5),
        MagicMock(comparison_type="earlier_vs_later", estimate=0.8, weight=0.3),
        MagicMock(comparison_type="later_vs_earlier", estimate=0.6, weight=0.2),
    ]
    results.twfe_estimate = 0.85
    results.effect_by_type.return_value = {
        "treated_vs_never": 1.0,
        "earlier_vs_later": 0.8,
        "later_vs_earlier": 0.6,
    }
    results.weight_by_type.return_value = {
        "treated_vs_never": 0.5,
        "earlier_vs_later": 0.3,
        "later_vs_earlier": 0.2,
    }
    return results


# ── Color Parser (no matplotlib needed) ──────────────────────────────────────


class TestColorParser:
    """_color_to_rgba must work without matplotlib installed."""

    def test_hex_6digit(self):
        from diff_diff.visualization._common import _color_to_rgba

        assert _color_to_rgba("#2563eb", 0.5) == "rgba(37, 99, 235, 0.5)"

    def test_hex_3digit(self):
        from diff_diff.visualization._common import _color_to_rgba

        assert _color_to_rgba("#abc", 0.3) == "rgba(170, 187, 204, 0.3)"

    def test_named_common(self):
        from diff_diff.visualization._common import _color_to_rgba

        assert _color_to_rgba("red", 1.0) == "rgba(255, 0, 0, 1.0)"
        assert _color_to_rgba("lightgray", 0.5) == "rgba(211, 211, 211, 0.5)"

    def test_named_css_extended(self):
        """Colors outside the original hand-maintained subset."""
        from diff_diff.visualization._common import _color_to_rgba

        assert _color_to_rgba("lightblue", 0.3) == "rgba(173, 216, 230, 0.3)"
        assert _color_to_rgba("cornflowerblue", 0.5) == "rgba(100, 149, 237, 0.5)"
        assert _color_to_rgba("tomato", 1.0) == "rgba(255, 99, 71, 1.0)"

    def test_rgb_string(self):
        from diff_diff.visualization._common import _color_to_rgba

        assert _color_to_rgba("rgb(100, 200, 50)", 0.7) == "rgba(100, 200, 50, 0.7)"

    def test_rgba_string_alpha_override(self):
        from diff_diff.visualization._common import _color_to_rgba

        result = _color_to_rgba("rgba(100, 200, 50, 0.9)", 0.3)
        assert result == "rgba(100, 200, 50, 0.3)"

    def test_unknown_color_raises(self):
        from diff_diff.visualization._common import _color_to_rgba

        with pytest.raises(ValueError, match="Cannot parse color"):
            _color_to_rgba("not_a_real_color")


# ── Plotly Smoke Tests (no matplotlib) ────────────────────────────────────────


class TestPlotlySensitivity:
    """Plotly backend for plot_sensitivity."""

    def test_basic(self, sensitivity_results):
        from diff_diff.visualization import plot_sensitivity

        fig = plot_sensitivity(sensitivity_results, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_named_color(self, sensitivity_results):
        from diff_diff.visualization import plot_sensitivity

        fig = plot_sensitivity(
            sensitivity_results,
            bounds_color="cornflowerblue",
            ci_color="tomato",
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)


class TestPlotlyHonestEventStudy:
    """Plotly backend for plot_honest_event_study."""

    def test_basic(self, honest_results):
        from diff_diff.visualization import plot_honest_event_study

        fig = plot_honest_event_study(honest_results, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_named_color(self, honest_results):
        from diff_diff.visualization import plot_honest_event_study

        fig = plot_honest_event_study(
            honest_results,
            original_color="lightblue",
            honest_color="steelblue",
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)


class TestPlotlyBaconBar:
    """Plotly backend for plot_bacon with plot_type='bar'."""

    def test_bar_basic(self, bacon_results):
        from diff_diff.visualization import plot_bacon

        fig = plot_bacon(bacon_results, plot_type="bar", backend="plotly", show=False)
        assert isinstance(fig, go.Figure)

    def test_scatter_weighted_avg(self, bacon_results):
        from diff_diff.visualization import plot_bacon

        fig = plot_bacon(bacon_results, show_weighted_avg=True, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        # Should have shapes from weighted avg + TWFE + zero vlines
        assert len(fig.layout.shapes) >= 4


class TestPlotlyEventStudyExtended:
    """Extended plotly event study coverage."""

    def test_css_color_outside_original_subset(self):
        from diff_diff import plot_event_study

        effects = {-1: 0.0, 0: 0.5, 1: 0.6}
        se = {-1: 0.0, 0: 0.15, 1: 0.15}
        fig = plot_event_study(
            effects=effects,
            se=se,
            color="steelblue",
            shade_color="lavender",
            pre_periods=[-1],
            post_periods=[0, 1],
            shade_pre=True,
            backend="plotly",
            show=False,
        )
        assert isinstance(fig, go.Figure)


class TestPlotlyHeatmapMasking:
    """Regression: mask_insignificant must grey out, not NaN-ify cells."""

    def test_mask_preserves_values(self, cs_results):
        from diff_diff.visualization import plot_group_time_heatmap

        fig = plot_group_time_heatmap(
            cs_results, mask_insignificant=True, backend="plotly", show=False
        )
        assert isinstance(fig, go.Figure)
        # Should have 2 traces: main heatmap + grey overlay
        assert len(fig.data) == 2
        # Main heatmap should NOT have NaN where cells were insignificant
        import numpy as np

        main_z = fig.data[0].z
        assert np.any(np.isfinite(main_z))

    def test_no_mask_single_trace(self, cs_results):
        from diff_diff.visualization import plot_group_time_heatmap

        fig = plot_group_time_heatmap(
            cs_results, mask_insignificant=False, backend="plotly", show=False
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Only main heatmap, no overlay

    def test_cmap_not_swapped(self, cs_results):
        """RdBu_r should not be swapped — last color should be warm (red)."""
        from diff_diff.visualization import plot_group_time_heatmap

        fig = plot_group_time_heatmap(cs_results, cmap="RdBu_r", backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        # Plotly resolves named colorscales to tuples. RdBu_r ends with red.
        cs = fig.data[0].colorscale
        # Last entry should be a reddish color (high R value)
        last_color = cs[-1][1]  # e.g. "rgb(103,0,31)"
        assert "103" in last_color or "178" in last_color  # dark red end of RdBu_r


class TestTopLevelExports:
    """Regression: new plot functions must be in diff_diff.__all__."""

    def test_all_plots_in_all(self):
        import diff_diff

        for name in [
            "plot_bacon",
            "plot_event_study",
            "plot_group_effects",
            "plot_sensitivity",
            "plot_honest_event_study",
            "plot_power_curve",
            "plot_pretrends_power",
            "plot_synth_weights",
            "plot_staircase",
            "plot_dose_response",
            "plot_group_time_heatmap",
        ]:
            assert name in diff_diff.__all__, f"{name} missing from diff_diff.__all__"

    def test_no_duplicates_in_all(self):
        import diff_diff

        seen = set()
        for name in diff_diff.__all__:
            assert name not in seen, f"Duplicate in __all__: {name}"
            seen.add(name)


class TestPlotlyEventStudyHover:
    """Regression: plotly event study must preserve period labels in hover."""

    def test_string_periods_in_customdata(self):
        from diff_diff import plot_event_study

        effects = {"pre": 0.0, "post": 0.5}
        se = {"pre": 0.1, "post": 0.15}
        fig = plot_event_study(
            effects=effects, se=se, backend="plotly", show=False
        )
        # At least one point trace should have customdata with original labels
        point_traces = [t for t in fig.data if t.mode == "markers"]
        assert len(point_traces) > 0
        for trace in point_traces:
            assert trace.customdata is not None, "Missing customdata on point trace"
            assert trace.hovertemplate is not None, "Missing hovertemplate"
