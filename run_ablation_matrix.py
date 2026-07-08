"""Run an ablation matrix for Hybrid Scheme 1.

This script orchestrates multiple `pipeline.py` runs (different combinations of
feature selection + adversarial augmentation) and aggregates the resulting
`metrics_attack_eval.csv` files into a single long-form CSV and a pivot summary.

Recommended usage (Windows / PowerShell):
    cd hybrid_scheme1
    ..\lab-ids-anta\Scripts\python.exe .\run_ablation_matrix.py \
        --dataset ..\lab-ids-anta-main\Dataset\normalized_data_2017.csv

Outputs are written under `--output-root` (defaults to results/ablation_YYYYMMDD).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class AblationRun:
    run_id: str
    group: str
    adversarial_method: str
    feature_selection: Optional[str]
    extra_args: List[str]


def _run_pipeline(python_exe: str, pipeline_path: Path, args: List[str]) -> None:
    cmd = [python_exe, str(pipeline_path)] + args
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def _read_attack_eval(csv_path: Path, run_meta: Dict[str, object]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for k, v in run_meta.items():
        df[k] = v
    return df


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a small DataFrame as a markdown table without extra deps.

    Pandas' `DataFrame.to_markdown()` requires the optional `tabulate` package.
    For reproducibility in minimal environments, we render a basic pipe table.
    """
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    def fmt_val(v: object) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, int)):
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, float):
                return float_fmt.format(v)
            return str(v)
        s = str(v)
        return s.replace("|", "\\|").replace("\n", "<br>")

    lines = [header, sep]
    for _, row in df.iterrows():
        vals = [fmt_val(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset = repo_root / "lab-ids-anta-main" / "Dataset" / "normalized_data_2017.csv"
    default_out_root = Path(__file__).resolve().parent / "results" / (
        "ablation_" + datetime.now().strftime("%Y%m%d")
    )

    p = argparse.ArgumentParser(description="Run ablation matrix experiments")
    p.add_argument("--dataset", type=Path, default=default_dataset, help="Path to CSV dataset")
    p.add_argument(
        "--output-root",
        type=Path,
        default=default_out_root,
        help="Root dir where each run's output will be stored",
    )
    p.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter to run pipeline.py (default: current interpreter)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a run folder already has metrics_attack_eval.csv",
    )

    # Shared hyperparams (keep these aligned across all runs)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--clip-value", type=float, default=3.0)
    p.add_argument("--eval-attacks", nargs="+", default=["fgsm", "pgd", "transfer_pgd"])
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--pgd-random-start", action="store_true", default=True)

    # Helper / surrogate DNN training
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--transfer-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--verbosity", type=int, default=0)

    # OWC-SAWN
    p.add_argument("--owc-epochs", type=int, default=100)

    # Feature selection
    p.add_argument("--max-features", type=int, default=20)
    p.add_argument("--fs-cv", type=int, default=3)
    p.add_argument(
        "--fs-max-samples",
        type=int,
        default=10000,
        help="Subsample size used only during feature selection (set 0/None to disable)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_root = args.output_root.resolve()
    python_exe = str(args.python.resolve())

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    output_root.mkdir(parents=True, exist_ok=True)

    pipeline_path = Path(__file__).resolve().parent / "pipeline.py"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"pipeline.py not found: {pipeline_path}")

    shared: List[str] = [
        "--dataset",
        str(dataset),
        "--epsilon",
        str(args.epsilon),
        "--clip-value",
        str(args.clip_value),
        "--eval-attacks",
        *list(args.eval_attacks),
        "--pgd-steps",
        str(args.pgd_steps),
        "--pgd-random-start",
        "--epochs",
        str(args.epochs),
        "--transfer-epochs",
        str(args.transfer_epochs),
        "--batch-size",
        str(args.batch_size),
        "--verbosity",
        str(args.verbosity),
        "--owc-epochs",
        str(args.owc_epochs),
        "--max-features",
        str(args.max_features),
        "--fs-cv",
        str(args.fs_cv),
    ]

    if args.fs_max_samples is not None and int(args.fs_max_samples) > 0:
        shared += ["--fs-max-samples", str(int(args.fs_max_samples))]

    runs: List[AblationRun] = [
        # a) No feature selection, no augmentation
        AblationRun(
            run_id="a_no_fs_no_aug",
            group="a",
            adversarial_method="none",
            feature_selection=None,
            extra_args=["--adversarial-method", "none"],
        ),
        # b) Feature selection only (forward)
        AblationRun(
            run_id="b_fs_forward_only",
            group="b",
            adversarial_method="none",
            feature_selection="forward",
            extra_args=[
                "--adversarial-method",
                "none",
                "--feature-selection",
                "forward",
            ],
        ),
        # c) FGSM augmentation only
        AblationRun(
            run_id="c_fgsm_only",
            group="c",
            adversarial_method="fgsm",
            feature_selection=None,
            extra_args=["--adversarial-method", "fgsm"],
        ),
        # d) OWC-SAWN augmentation only
        AblationRun(
            run_id="d_owc_only",
            group="d",
            adversarial_method="owc-sawn",
            feature_selection=None,
            extra_args=["--adversarial-method", "owc-sawn"],
        ),
        # e) FGSM + OWC-SAWN (main method)
        AblationRun(
            run_id="e_fgsm_owc",
            group="e",
            adversarial_method="both",
            feature_selection=None,
            extra_args=["--adversarial-method", "both"],
        ),
        # f) Feature selection strategy comparison under the main method
        AblationRun(
            run_id="f_fs_forward_main",
            group="f",
            adversarial_method="both",
            feature_selection="forward",
            extra_args=[
                "--adversarial-method",
                "both",
                "--feature-selection",
                "forward",
            ],
        ),
        AblationRun(
            run_id="f_fs_backward_main",
            group="f",
            adversarial_method="both",
            feature_selection="backward",
            extra_args=[
                "--adversarial-method",
                "both",
                "--feature-selection",
                "backward",
            ],
        ),
        AblationRun(
            run_id="f_fs_correlation_main",
            group="f",
            adversarial_method="both",
            feature_selection="correlation",
            extra_args=[
                "--adversarial-method",
                "both",
                "--feature-selection",
                "correlation",
            ],
        ),
        AblationRun(
            run_id="f_fs_importance_main",
            group="f",
            adversarial_method="both",
            feature_selection="importance",
            extra_args=[
                "--adversarial-method",
                "both",
                "--feature-selection",
                "importance",
            ],
        ),
    ]

    # Execute runs
    for r in runs:
        run_dir = output_root / r.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        attack_eval_csv = run_dir / "metrics_attack_eval.csv"
        if attack_eval_csv.exists() and not args.force:
            print(f"\n[SKIP] {r.run_id}: metrics_attack_eval.csv already exists")
            continue

        cmd_args = shared + r.extra_args + ["--output-dir", str(run_dir)]
        _run_pipeline(python_exe, pipeline_path, cmd_args)

        if not attack_eval_csv.exists():
            raise RuntimeError(f"Expected output missing: {attack_eval_csv}")

    # Aggregate results
    long_rows: List[pd.DataFrame] = []
    for r in runs:
        run_dir = output_root / r.run_id
        csv_path = run_dir / "metrics_attack_eval.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing metrics_attack_eval.csv for run: {r.run_id}")
            continue

        meta = {
            "run_id": r.run_id,
            "group": r.group,
            "dataset": str(dataset),
            "adversarial_method": r.adversarial_method,
            "feature_selection": r.feature_selection or "",
            "output_dir": str(run_dir),
        }
        long_rows.append(_read_attack_eval(csv_path, meta))

    if not long_rows:
        raise RuntimeError("No attack eval CSVs found; nothing to aggregate")

    long_df = pd.concat(long_rows, ignore_index=True)

    long_csv = output_root / "ablation_attack_eval_long.csv"
    long_df.to_csv(long_csv, index=False)

    # Pivot summary: F1 per (run, model, train_regime) across attacks
    pivot = long_df.pivot_table(
        index=["group", "run_id", "adversarial_method", "feature_selection", "model", "train_regime"],
        columns=["attack"],
        values=["f1", "accuracy", "auc"],
        aggfunc="first",
    )

    # Flatten columns: (metric, attack) -> metric__attack
    pivot.columns = [f"{metric}__{attack}" for metric, attack in pivot.columns]
    pivot = pivot.reset_index().sort_values(["group", "run_id", "model", "train_regime"])

    pivot_csv = output_root / "ablation_summary_pivot.csv"
    pivot.to_csv(pivot_csv, index=False)

    # Small markdown summary for quick inspection (F1 only)
    f1_cols = [c for c in pivot.columns if c.startswith("f1__")]
    md_df = pivot[["group", "run_id", "model", "train_regime"] + f1_cols]
    md = _to_markdown_table(md_df)
    md_path = output_root / "ablation_summary_f1.md"
    _safe_write_text(md_path, "# Ablation Summary (F1)\n\n" + md + "\n")

    # Reviewer-friendly compact tables (SGDClassifier)
    def pick_row(run_id: str, train_regime: str) -> pd.Series:
        sub = pivot[
            (pivot["run_id"] == run_id)
            & (pivot["model"] == "SGDClassifier")
            & (pivot["train_regime"] == train_regime)
        ]
        if len(sub) != 1:
            raise RuntimeError(
                f"Expected exactly 1 row for run_id={run_id}, model=SGDClassifier, train_regime={train_regime}; got {len(sub)}"
            )
        return sub.iloc[0]

    core_a_to_e = []
    for group, run_id, regime in [
        ("a", "a_no_fs_no_aug", "baseline"),
        ("b", "b_fs_forward_only", "baseline"),
        ("c", "c_fgsm_only", "adv_train"),
        ("d", "d_owc_only", "adv_train"),
        ("e", "e_fgsm_owc", "adv_train"),
    ]:
        r = pick_row(run_id, regime)
        core_a_to_e.append(
            {
                "group": group,
                "run_id": run_id,
                "train_regime_used": regime,
                "adversarial_method": r.get("adversarial_method", ""),
                "feature_selection": r.get("feature_selection", ""),
                "f1__clean": r.get("f1__clean"),
                "f1__fgsm": r.get("f1__fgsm"),
                "f1__pgd": r.get("f1__pgd"),
                "f1__transfer_pgd": r.get("f1__transfer_pgd"),
            }
        )
    core_a_to_e_df = pd.DataFrame(core_a_to_e)
    core_a_to_e_csv = output_root / "ablation_core_a_to_e_sgd.csv"
    core_a_to_e_df.to_csv(core_a_to_e_csv, index=False)
    core_a_to_e_md = output_root / "ablation_core_a_to_e_sgd.md"
    _safe_write_text(
        core_a_to_e_md,
        "# Ablation Core (a–e) — SGDClassifier\n\n" + _to_markdown_table(core_a_to_e_df) + "\n",
    )

    fs_rows = []
    for run_id in [
        "f_fs_forward_main",
        "f_fs_backward_main",
        "f_fs_correlation_main",
        "f_fs_importance_main",
    ]:
        r = pick_row(run_id, "adv_train")
        fs_rows.append(
            {
                "run_id": run_id,
                "feature_selection": r.get("feature_selection", ""),
                "f1__clean": r.get("f1__clean"),
                "f1__fgsm": r.get("f1__fgsm"),
                "f1__pgd": r.get("f1__pgd"),
                "f1__transfer_pgd": r.get("f1__transfer_pgd"),
            }
        )
    fs_df = pd.DataFrame(fs_rows)
    fs_csv = output_root / "ablation_fs_strategies_main_sgd.csv"
    fs_df.to_csv(fs_csv, index=False)
    fs_md = output_root / "ablation_fs_strategies_main_sgd.md"
    _safe_write_text(
        fs_md,
        "# Feature Selection Strategies (f) under Main Method — SGDClassifier (adv_train)\n\n"
        + _to_markdown_table(fs_df)
        + "\n",
    )

    # Save run manifest
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "output_root": str(output_root),
        "python": python_exe,
        "shared_args": shared,
        "runs": [
            {
                "run_id": r.run_id,
                "group": r.group,
                "adversarial_method": r.adversarial_method,
                "feature_selection": r.feature_selection,
                "extra_args": r.extra_args,
            }
            for r in runs
        ],
    }
    (output_root / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n✓ Ablation runs complete")
    print("Long-form:", long_csv)
    print("Pivot summary:", pivot_csv)
    print("Markdown F1:", md_path)
    print("Core (a-e, SGD):", core_a_to_e_csv)
    print("FS strategies (f, SGD):", fs_csv)


if __name__ == "__main__":
    main()
