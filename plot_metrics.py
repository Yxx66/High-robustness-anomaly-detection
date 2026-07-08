from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# Mapping keeps scenario identifiers consistent between data and figure labels.
SCENARIO_LABELS: Dict[str, str] = {
    "baseline_clean_train_clean_test": "Clean train → Clean test",
    "baseline_clean_train_adv_test": "Clean train → Adv test",
    "adv_train_adv_test": "Adv train → Adv test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scenario comparison metrics.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/metrics.csv"),
        help="Path to metrics CSV produced by pipeline.py",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["accuracy", "precision", "recall", "f1", "auc"],
        help="Metric column to plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/scenario_comparison.png"),
        help="Where to save the generated figure",
    )
    parser.add_argument(
        "--scenario-order",
        nargs="*",
        default=[
            "baseline_clean_train_clean_test",
            "baseline_clean_train_adv_test",
            "adv_train_adv_test",
        ],
        help="Subset and order of scenarios to keep in the figure",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.metrics.exists():
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")
    missing = [s for s in args.scenario_order if s not in SCENARIO_LABELS]
    if missing:
        known = ", ".join(SCENARIO_LABELS.keys())
        raise ValueError(f"Unknown scenarios {missing}. Known values: {known}")


def load_metrics(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.metrics)
    required_columns = {"model", "scenario", args.metric}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        cols = ", ".join(sorted(missing_cols))
        raise ValueError(f"Metrics file missing columns: {cols}")
    df = df[df["scenario"].isin(args.scenario_order)]
    if df.empty:
        raise ValueError("No rows remain after filtering by scenario order")
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
    # Preserve requested ordering for consistent plotting.
    df["scenario_label"] = pd.Categorical(
        df["scenario_label"],
        categories=[SCENARIO_LABELS[s] for s in args.scenario_order],
        ordered=True,
    )
    return df


def plot(df: pd.DataFrame, args: argparse.Namespace) -> None:
    pivot = (
        df[["scenario_label", "model", args.metric]]
        .pivot(index="scenario_label", columns="model", values=args.metric)
        .sort_index()
    )
    models: List[str] = list(pivot.columns)
    x = range(len(pivot.index))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, model in enumerate(models):
        offset = (idx - (len(models) - 1) / 2) * width
        positions = [pos + offset for pos in x]
        values = pivot[model].to_list()
        bars = ax.bar(positions, values, width=width, label=model)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel(args.metric.upper())
    ax.set_ylim(0, 1.05)
    ax.set_title("Scenario Comparison")
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    df = load_metrics(args)
    plot(df, args)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
