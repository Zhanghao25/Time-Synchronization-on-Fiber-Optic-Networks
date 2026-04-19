#!/usr/bin/env python3
"""
Build the canonical readable outputs for:
- Figure 5 (City 1)
- Figure S.5 (City 2)
- Figure S.6 (City 3)

Default outputs are intentionally minimal:
- paper_outputs/figures/readable/{figure5_city1,figure_s5_city2,figure_s6_city3}.{png,pdf}
- paper_outputs/figures/readable/*_plot_data.{csv,xlsx}

This script follows the notebook plotting style exactly and only replaces
manual notebook values with reproducible metric summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "lur_figures"
READABLE_DIR = ROOT / "paper_outputs" / "figures" / "readable"
CITY_META = {
    "City 1": {
        "figure_id": "Figure 5",
        "stem": "figure5_city1",
        "metrics_xlsx": METRICS_DIR / "lur_city1_metrics.xlsx",
    },
    "City 2": {
        "figure_id": "Figure S.5",
        "stem": "figure_s5_city2",
        "metrics_xlsx": METRICS_DIR / "lur_city2_metrics.xlsx",
    },
    "City 3": {
        "figure_id": "Figure S.6",
        "stem": "figure_s6_city3",
        "metrics_xlsx": METRICS_DIR / "lur_city3_metrics.xlsx",
    },
}

SUMMARY_COLUMNS = [
    "Figure",
    "City",
    "alpha",
    "alpha_label",
    "experiments_completed",
    "P_err_LUR",
    "P_err_LUR_std",
    "R_asy_LUR",
    "R_asy_LUR_std",
    "R_asy_nonLUR",
    "R_asy_nonLUR_std",
    "UHCE_Edges_Count",
    "UHCE_Edges_Count_std",
    "HCE_Edges_Count",
    "HCE_Edges_Count_std",
    "SHCE_Edges_Count",
    "SHCE_Edges_Count_std",
    "UHCE_Confidence",
    "UHCE_Confidence_std",
    "HCE_Confidence",
    "HCE_Confidence_std",
    "SHCE_Confidence",
    "SHCE_Confidence_std",
]


def load_city_summary(city_name: str, metrics_dir: Path) -> pd.DataFrame:
    meta = dict(CITY_META[city_name])
    xlsx_path = metrics_dir / meta["metrics_xlsx"].name
    summary = pd.read_excel(xlsx_path, sheet_name="Summary").copy().sort_values("alpha").reset_index(drop=True)
    summary.insert(0, "City", city_name)
    summary.insert(0, "Figure", meta["figure_id"])
    return summary[SUMMARY_COLUMNS]


def build_plot_figure(city_table: pd.DataFrame) -> plt.Figure:
    proportions = [f"{int(round(value * 100))}%" for value in city_table["alpha"].tolist()]
    uhces_numbers = city_table["UHCE_Edges_Count"].tolist()
    hces_numbers = city_table["HCE_Edges_Count"].tolist()
    shces_numbers = city_table["SHCE_Edges_Count"].tolist()
    uhces_confidence = city_table["UHCE_Confidence"].tolist()
    hces_confidence = city_table["HCE_Confidence"].tolist()
    shces_confidence = city_table["SHCE_Confidence"].tolist()
    y1 = [value * 100 for value in city_table["P_err_LUR"].tolist()]
    y2 = [value * 100 for value in city_table["R_asy_LUR"].tolist()]
    y3 = [value * 100 for value in city_table["R_asy_nonLUR"].tolist()]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.5,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    ax3, ax4, ax1, ax2 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96, wspace=0.18, hspace=0.32)

    x = list(range(len(proportions)))
    colors = {
        "UHCE": "#1f77b4",
        "HCE": "#ff7f0e",
        "SHCE": "#2ca02c",
        "P_err": "#821fb4",
        "R_asy": "#ffdb0e",
        "R_non": "#a04d2c",
    }

    ax3.text(0.5, -0.16, "(a)", transform=ax3.transAxes, ha="center", fontsize=20)
    ax4.text(0.5, -0.16, "(b)", transform=ax4.transAxes, ha="center", fontsize=20)
    ax1.text(0.5, -0.18, "(c)", transform=ax1.transAxes, ha="center", fontsize=20)
    ax2.text(0.5, -0.18, "(d)", transform=ax2.transAxes, ha="center", fontsize=20)

    width = 0.25
    ax1.bar([item - width for item in x], uhces_numbers, width, label="UHCE", color=colors["UHCE"], edgecolor="white")
    ax1.bar(x, hces_numbers, width, label="HCE", color=colors["HCE"], edgecolor="white")
    ax1.bar([item + width for item in x], shces_numbers, width, label="SHCE", color=colors["SHCE"], edgecolor="white")
    ax1.set_xlabel("α", fontsize=16)
    ax1.set_ylabel("Number of edges", fontsize=16)
    ax1.set_ylim(0, max(max(uhces_numbers), max(hces_numbers), max(shces_numbers)) * 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(proportions, fontsize=14)
    ax1.legend(loc="upper left", frameon=True, fontsize=12)
    ax1.set_axisbelow(True)

    ax2.plot(x, uhces_confidence, marker="o", label="UHCE", color=colors["UHCE"], markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax2.plot(x, hces_confidence, marker="s", label="HCE", color=colors["HCE"], markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax2.plot(x, shces_confidence, marker="^", label="SHCE", color=colors["SHCE"], markersize=7, markerfacecolor="white", markeredgewidth=2)
    min_conf = min(min(uhces_confidence), min(hces_confidence), min(shces_confidence)) * 0.98
    ax2.set_ylim(min_conf, 1.005)
    ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax2.set_xlabel("α", fontsize=16)
    ax2.set_ylabel("Precision", fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(proportions, fontsize=14)
    ax2.legend(loc="lower left", ncol=1, frameon=True, fontsize=12, title_fontsize=11)
    ax2.set_axisbelow(True)

    ax3.plot(x, y1, "o-", label=r"$P_{\mathrm{err}}^{\mathrm{LUR}}$", color=colors["P_err"], markersize=7)
    ax3.set_ylim(95, 100.5)
    ax3.yaxis.set_major_locator(MultipleLocator(2))
    ax3.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax3.set_xlabel(r"$\alpha$", fontsize=16)
    ax3.set_ylabel(r"$\beta$", fontsize=16, rotation=360)
    ax3.set_xticks(x)
    ax3.set_xticklabels(proportions, fontsize=14)
    ax3.set_axisbelow(True)

    ax4.plot(x, y2, "s-", label="LUR", color=colors["R_asy"], markersize=7)
    ax4.plot(x, y3, "^-", label="Non-LUR", color=colors["R_non"], markersize=7)
    ax4.set_ylim(0, 40)
    ax4.yaxis.set_major_locator(MultipleLocator(10))
    ax4.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax4.set_xlabel(r"$\alpha$", fontsize=16)
    ax4.set_ylabel("Proportion of asymmetric edges", fontsize=16)
    ax4.set_xticks(x)
    ax4.set_xticklabels(proportions, fontsize=14)
    ax4.legend(loc="upper left", frameon=True, fontsize=12, title_fontsize=11)
    ax4.set_axisbelow(True)
    return fig


def export_readable_outputs(city_tables: Dict[str, pd.DataFrame], readable_dir: Path) -> None:
    readable_dir.mkdir(parents=True, exist_ok=True)
    for city_name, city_table in city_tables.items():
        stem = CITY_META[city_name]["stem"]
        city_table.to_csv(readable_dir / f"{stem}_plot_data.csv", index=False)
        city_table.to_excel(readable_dir / f"{stem}_plot_data.xlsx", index=False)
        fig = build_plot_figure(city_table)
        fig.savefig(readable_dir / f"{stem}.png", format="png", dpi=900, bbox_inches="tight")
        fig.savefig(readable_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def build_all(metrics_dir: Path, readable_dir: Path) -> None:
    city_tables = {city: load_city_summary(city, metrics_dir) for city in CITY_META}
    export_readable_outputs(city_tables, readable_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 5 / Figure S.5 / Figure S.6 from canonical metrics.")
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=READABLE_DIR, help="Readable figure output directory.")
    args = parser.parse_args()

    build_all(args.metrics_dir, args.output_dir)


if __name__ == "__main__":
    main()
