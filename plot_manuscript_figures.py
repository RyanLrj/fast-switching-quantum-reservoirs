"""Generate the four publication figures used by the manuscript.

All data panels are deterministic transformations of the checked-in CSV files.
Outputs are written as vector PDF/SVG and 600 dpi PNG.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manuscript_figures"
OUT.mkdir(exist_ok=True)

COLORS = {0.0: "#0072B2", 0.4: "#D55E00", 0.8: "#009E73"}
MARKERS = {0.0: "o", 0.4: "s", 0.8: "^"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_numeric_csv(filename: str) -> dict[str, np.ndarray]:
    with (ROOT / filename).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    numeric: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            numeric[key] = np.asarray([float(row[key]) for row in rows])
        except ValueError:
            continue
    return numeric


def save_all(fig: plt.Figure, stem: str) -> None:
    pdf_metadata = {
        "Title": stem.replace("_", " ").title(),
        "Author": "Generated from manuscript data",
        "Subject": "Reproducible scientific figure",
    }
    svg_metadata = {
        "Title": stem.replace("_", " ").title(),
        "Description": "Reproducible scientific figure generated from manuscript data",
    }
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata=pdf_metadata)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", metadata=svg_metadata)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def arrow(ax, start, end, color="#333333", style="-|>", lw=1.6, ls="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(patch)
    return patch


def rounded_box(ax, xy, width, height, face, edge, title, subtitle=None):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.6,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(x + width / 2, y + height * 0.60, title, ha="center", va="center", weight="bold")
    if subtitle:
        ax.text(x + width / 2, y + height * 0.28, subtitle, ha="center", va="center", fontsize=8)
    return box


def figure1_model_schematic() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    hidden = FancyBboxPatch(
        (0.35, 0.25),
        5.1,
        3.65,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        facecolor="#F4F4F4",
        edgecolor="#777777",
        linewidth=1.2,
        linestyle="--",
    )
    ax.add_patch(hidden)
    ax.text(0.58, 3.63, "Hidden environment", color="#555555", weight="bold")
    ax.text(0.58, 0.48, "reservoir label is never observed", color="#666666", fontsize=8)

    rounded_box(ax, (0.75, 2.25), 1.75, 0.95, "#FADBD8", "#C0392B", "Hot reservoir", r"$\beta_h,\gamma_h$")
    rounded_box(ax, (0.75, 0.85), 1.75, 0.95, "#D6EAF8", "#2471A3", "Cold reservoir", r"$\beta_c,\gamma_c$")

    switch = Circle((4.0, 2.02), 0.72, facecolor="#FFF2CC", edgecolor="#A67C00", linewidth=1.6)
    ax.add_patch(switch)
    ax.text(4.0, 2.19, "hidden label", ha="center", va="center", fontsize=8)
    ax.text(4.0, 1.86, r"$\alpha(t)$", ha="center", va="center", fontsize=12)
    ax.text(4.0, 1.18, r"Markov switching at rate $k$", ha="center", fontsize=8)

    arrow(ax, (2.50, 2.72), (3.35, 2.28), color="#C0392B")
    arrow(ax, (2.50, 1.32), (3.35, 1.78), color="#2471A3")
    arrow(ax, (3.48, 2.67), (3.48, 1.36), color="#777777", style="<->", lw=1.2, ls="--")

    qubit = Circle((6.72, 2.02), 0.70, facecolor="#E8DAEF", edgecolor="#6C3483", linewidth=1.8)
    ax.add_patch(qubit)
    ax.text(6.72, 2.20, "Qubit", ha="center", va="center", weight="bold")
    ax.text(6.72, 1.83, r"$\rho(t)$", ha="center", va="center", fontsize=11)
    arrow(ax, (4.74, 2.02), (5.98, 2.02), color="#333333", lw=1.8)
    ax.text(5.36, 2.30, r"$\mathcal{L}_{\alpha(t)}$", ha="center", fontsize=10)

    rounded_box(ax, (8.25, 1.47), 2.10, 1.10, "#D5F5E3", "#1E8449", "System measurement", r"fixed protocol $\{M_y\}$")
    arrow(ax, (7.44, 2.02), (8.24, 2.02), color="#333333", lw=1.8)
    ax.text(9.30, 0.98, "system-only observer", ha="center", weight="bold", color="#1E8449")

    ax.plot([5.65, 5.65], [0.35, 3.70], color="#888888", linestyle=":", linewidth=1.2)
    ax.text(5.49, 3.48, "unobserved", rotation=90, va="top", ha="right", fontsize=8, color="#666666")
    ax.text(5.81, 3.48, "observed", rotation=90, va="top", ha="left", fontsize=8, color="#1E8449")

    ax.text(
        8.00,
        3.45,
        r"Fast limit: $\mathcal{L}_{\mathrm{eff}}=\sum_\alpha p_\alpha\mathcal{L}_\alpha$",
        ha="center",
        va="center",
        fontsize=10,
    )
    save_all(fig, "figure1_model_schematic")


def grouped(data: dict[str, np.ndarray]):
    for omega in sorted(COLORS):
        mask = np.isclose(data["Omega"], omega)
        order = np.argsort(data["switch_rate"][mask])
        yield omega, data["switch_rate"][mask][order], mask, order


def style_axis(ax):
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.6)
    ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
    ax.tick_params(direction="in", which="both", top=True, right=True)


def figure2_independent_scaling() -> None:
    data = read_numeric_csv("fourth_round_detectability_scan.csv")
    columns = [
        "absolute_probability_difference",
        "bernoulli_KL_nats_per_shot",
        "shots_for_log_likelihood_log20",
    ]
    ylabels = [r"$|p_k-p_{\mathrm{eff}}|$", r"$D_{\mathrm{KL}}$ per shot", r"$\log(20)/D_{\mathrm{KL}}$"]
    slopes = [-1, -2, 2]
    panels = ["(a)", "(b)", "(c)"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35))

    for ax, column, ylabel, slope, panel in zip(axes, columns, ylabels, slopes, panels):
        for omega, k, mask, order in grouped(data):
            values = data[column][mask][order]
            ax.loglog(
                k,
                values,
                color=COLORS[omega],
                marker=MARKERS[omega],
                markersize=3.2,
                markevery=8,
                label=rf"$\Omega={omega:g}$",
            )

        refmask = np.isclose(data["Omega"], 0.0) & (data["switch_rate"] >= 100)
        kref = data["switch_rate"][refmask]
        yref = data[column][refmask]
        order = np.argsort(kref)
        kref, yref = kref[order], yref[order]
        guide = yref[0] * (kref / kref[0]) ** slope
        ax.loglog(kref, guide, "--", color="black", linewidth=1.0)
        ax.text(0.69, 0.14, rf"$k^{{{slope}}}$", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel(r"Switching rate $k$")
        ax.set_ylabel(ylabel)
        ax.text(0.04, 0.94, panel, transform=ax.transAxes, va="top", weight="bold")
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    save_all(fig, "figure2_independent_scaling")


def figure3_entropy_information_separation() -> None:
    data = read_numeric_csv("fourth_round_detectability_scan.csv")
    limits = {0.0: 0.132388, 0.4: 0.134467, 0.8: 0.139436}
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.0), sharex=True)

    for omega, k, mask, order in grouped(data):
        entropy = data["stationary_hidden_entropy_rate"][mask][order]
        kl = data["bernoulli_KL_nats_per_shot"][mask][order]
        axes[0].semilogx(
            k, entropy, color=COLORS[omega], marker=MARKERS[omega], markersize=3.2,
            markevery=8, label=rf"$\Omega={omega:g}$"
        )
        axes[0].axhline(limits[omega], color=COLORS[omega], linestyle="--", linewidth=0.9, alpha=0.75)
        axes[1].loglog(
            k, kl, color=COLORS[omega], marker=MARKERS[omega], markersize=3.2, markevery=8
        )

    refmask = np.isclose(data["Omega"], 0.0) & (data["switch_rate"] >= 100)
    kref = data["switch_rate"][refmask]
    yref = data["bernoulli_KL_nats_per_shot"][refmask]
    order = np.argsort(kref)
    kref, yref = kref[order], yref[order]
    axes[1].loglog(kref, yref[0] * (kref / kref[0]) ** -2, "--", color="black", linewidth=1.0)
    axes[1].text(0.77, 0.17, r"$k^{-2}$", transform=axes[1].transAxes)

    axes[0].set_ylabel(r"$\Delta\Sigma(k)$")
    axes[1].set_ylabel(r"$D_{\mathrm{KL}}$ per shot")
    axes[1].set_xlabel(r"Switching rate $k$")
    axes[0].text(0.025, 0.92, "(a)", transform=axes[0].transAxes, weight="bold")
    axes[1].text(0.025, 0.92, "(b)", transform=axes[1].transAxes, weight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="lower right")
    for ax in axes:
        style_axis(ax)
    fig.tight_layout(h_pad=0.7)
    save_all(fig, "figure3_entropy_information_separation")


def figure4_no_reset_coefficient() -> None:
    data = read_numeric_csv("fifth_round_trajectory_kl_scan.csv")
    theory = {0.0: 0.008182169574, 0.4: 0.007002264462, 0.8: 0.004322673954}
    fig, ax = plt.subplots(figsize=(5.6, 4.1))

    for omega, k, mask, order in grouped(data):
        rate = data["trajectory_KL_nats_per_measurement"][mask][order]
        stderr = data["KL_batch_standard_error"][mask][order]
        ax.errorbar(
            k,
            k**2 * rate,
            yerr=k**2 * stderr,
            color=COLORS[omega],
            marker=MARKERS[omega],
            markersize=4.0,
            linewidth=1.4,
            capsize=2.0,
            label=rf"$\Omega={omega:g}$",
        )
        ax.axhline(theory[omega], color=COLORS[omega], linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel(r"Switching rate $k$")
    ax.set_ylabel(r"Scaled trajectory information $k^2d_k$")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    save_all(fig, "figure4_no_reset_coefficient")


if __name__ == "__main__":
    figure1_model_schematic()
    figure2_independent_scaling()
    figure3_entropy_information_separation()
    figure4_no_reset_coefficient()
    print(f"Wrote figures to {OUT}")
