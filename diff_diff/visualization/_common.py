"""Shared utilities for the visualization subpackage."""


def _require_matplotlib():
    """Lazy import matplotlib with clear error message.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module.
    """
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. " "Install it with: pip install matplotlib"
        )


def _require_plotly():
    """Lazy import plotly with clear error message.

    Returns
    -------
    module
        The ``plotly.graph_objects`` module.
    """
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install with: pip install diff-diff[plotly]"
        )


def _plotly_default_layout(fig, *, title=None, xlabel=None, ylabel=None, show_legend=True):
    """Apply standard plotly layout settings.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to configure.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    show_legend : bool, default=True
        Whether to show the legend.
    """
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=show_legend,
        template="plotly_white",
        font=dict(size=12),
        margin=dict(l=60, r=30, t=50, b=50),
    )


def _hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to rgba string for plotly.

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., ``"#2563eb"``).
    alpha : float, default=1.0
        Opacity value between 0 and 1.

    Returns
    -------
    str
        An ``rgba(r, g, b, a)`` string.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# Default color constants
DEFAULT_BLUE = "#2563eb"
DEFAULT_RED = "#dc2626"
DEFAULT_GREEN = "#22c55e"
DEFAULT_GRAY = "#6b7280"
DEFAULT_DARK = "#1f2937"
DEFAULT_SHADE = "#f0f0f0"
