"""Shared manuscript plotting theme and linear-fit helper.

Every figure notebook applies the same rcParams theme via ``set_paper_theme``.
The only variation across figures is the tick line width (0.5 pt default,
0.3 pt on some panels) and ``axes.unicode_minus`` (disabled where tick labels
contain '-'), so both are exposed as arguments.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist


PAPER_THEME = {
    "pdf.use14corefonts": True,
    "font.family": "Helvetica",
    "figure.constrained_layout.use": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.5,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6,
    "font.size": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
}


def set_paper_theme(tick_width=0.5, unicode_minus=None):
    """Apply the shared theme. ``unicode_minus=False`` renders '-' in tick
    labels with the PDF core font instead of the unicode minus glyph."""
    theme = dict(PAPER_THEME)
    theme["xtick.major.width"] = tick_width
    theme["ytick.major.width"] = tick_width
    if unicode_minus is not None:
        theme["axes.unicode_minus"] = unicode_minus
    plt.rcParams.update(theme)


def plot_linear_fit_ci(ax, x, y, color="black"):
    """Linear fit with its 95% confidence band (Figure6_2 / Figure6_4 style)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    dof = len(x) - 2
    s = np.sqrt(np.sum((y - y_est) ** 2) / dof)
    std_err = s * np.sqrt(1 / len(x)
                          + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    y_err = t_dist.ppf(0.975, dof) * std_err
    order = np.argsort(y_est)
    ax.plot(x[order], y_est[order], linewidth=0.5, color=color)
    ax.fill_between(x[order], y_est[order] - y_err[order], y_est[order] + y_err[order],
                    color=color, alpha=0.4, edgecolor="None")
