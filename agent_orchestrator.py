"""Minimal agent-style orchestrator for Hybrid Scheme 1.

This module wraps the existing IDS workflow into a small task planner with
tool-style execution steps. It is intentionally lightweight: the goal is to
turn the project into an agentic demo without rewriting the core research
pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class AgentStep:
    """One executable tool call in the agent plan."""

    name: str
    description: str
    command: List[str]


class HybridIDSAgent:
    """Rule-based agent that plans and executes IDS analysis tasks."""

    def __init__(self, repo_dir: Path, dataset: Path, output_dir: Path) -> None:
        self.repo_dir = repo_dir.resolve()
        self.dataset = dataset.resolve()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def infer_intent(self, task: str) -> Dict[str, bool]:
        text = task.lower()
        return {
            "feature_selection": bool(re.search(r"feature|特征", text)),
            "owc": bool(re.search(r"owc|gan|生成对抗|adversarial network", text)),
            "attack_eval": bool(re.search(r"robust|attack|对抗|鲁棒|评估", text)),
            "visualize": bool(re.search(r"plot|chart|图|可视化|report|报告", text)),
        }

    def profile_dataset(self) -> Dict[str, object]:
        df = pd.read_csv(self.dataset)
        profile = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_values": int(df.isna().sum().sum()),
            "numeric_columns": int(df.select_dtypes(include=["number"]).shape[1]),
        }
        if "Label" in df.columns:
            label_counts = df["Label"].astype(str).value_counts().head(10).to_dict()
            profile["label_distribution"] = {str(k): int(v) for k, v in label_counts.items()}
        return profile

    def build_plan(
        self,
        task: str,
        feature_selection: Optional[str] = None,
        adversarial_method: Optional[str] = None,
        max_features: int = 20,
    ) -> Dict[str, object]:
        intent = self.infer_intent(task)
        steps: List[AgentStep] = []

        steps.append(
            AgentStep(
                name="profile_dataset",
                description="Inspect dataset shape, label balance, and missing values.",
                command=[sys.executable, "-c", "from agent_orchestrator import _profile_entry; _profile_entry()"],
            )
        )

        if intent["feature_selection"] or feature_selection:
            fs_method = feature_selection or "compare"
            steps.append(
                AgentStep(
                    name="feature_selection",
                    description=f"Run feature selection using {fs_method}.",
                    command=[
                        sys.executable,
                        "evaluate_feature_selection.py",
                        "--dataset",
                        str(self.dataset),
                        "--max-features",
                        str(max_features),
                    ] if fs_method == "compare" else [
                        sys.executable,
                        "pipeline.py",
                        "--dataset",
                        str(self.dataset),
                        "--feature-selection",
                        fs_method,
                        "--max-features",
                        str(max_features),
                    ],
                )
            )

        if intent["owc"] or adversarial_method == "owc-sawn":
            steps.append(
                AgentStep(
                    name="owc_sawn_training",
                    description="Train OWC-SAWN adversarial generator and evaluate the pipeline.",
                    command=[
                        sys.executable,
                        "pipeline.py",
                        "--dataset",
                        str(self.dataset),
                        "--adversarial-method",
                        "owc-sawn",
                    ],
                )
            )

        if intent["attack_eval"] or adversarial_method in {"fgsm", "both"}:
            chosen_method = adversarial_method or "fgsm"
            steps.append(
                AgentStep(
                    name="robustness_evaluation",
                    description=f"Train adversarially with {chosen_method} and compare clean/attack scenarios.",
                    command=[
                        sys.executable,
                        "pipeline.py",
                        "--dataset",
                        str(self.dataset),
                        "--adversarial-method",
                        chosen_method,
                    ],
                )
            )

        if intent["visualize"]:
            steps.append(
                AgentStep(
                    name="report_generation",
                    description="Generate metrics tables and charts from the latest results.",
                    command=[sys.executable, "generate_visual_tables.py"],
                )
            )

        if not steps:
            steps.append(
                AgentStep(
                    name="baseline_pipeline",
                    description="Run the default IDS pipeline as the fallback action.",
                    command=[sys.executable, "pipeline.py", "--dataset", str(self.dataset)],
                )
            )

        return {
            "task": task,
            "dataset": str(self.dataset),
            "output_dir": str(self.output_dir),
            "intent": intent,
            "steps": [asdict(step) for step in steps],
        }

    def execute(self, plan: Dict[str, object]) -> Dict[str, object]:
        execution_log: List[Dict[str, object]] = []
        for raw_step in plan["steps"]:
            step = AgentStep(**raw_step)
            print(f"\n[{step.name}] {step.description}")
            completed = subprocess.run(
                step.command,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
            )
            execution_log.append(
                {
                    "name": step.name,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                }
            )
            if completed.returncode != 0:
                break
        return {"plan": plan, "execution": execution_log}

    def write_report(self, plan: Dict[str, object], execution: Optional[Dict[str, object]] = None) -> Path:
        report_path = self.output_dir / "agent_report.md"
        lines: List[str] = []
        lines.append("# Hybrid IDS Agent Report")
        lines.append("")
        lines.append(f"- Task: {plan['task']}")
        lines.append(f"- Dataset: {plan['dataset']}")
        lines.append(f"- Output dir: {plan['output_dir']}")
        lines.append("")
        lines.append("## Planned Steps")
        for idx, raw_step in enumerate(plan["steps"], start=1):
            lines.append(f"{idx}. {raw_step['name']}: {raw_step['description']}")
        lines.append("")
        lines.append("## Dataset Profile")
        try:
            profile = self.profile_dataset()
            lines.append(json.dumps(profile, indent=2, ensure_ascii=False))
        except Exception as exc:  # pragma: no cover - report should remain usable
            lines.append(f"Failed to profile dataset: {exc}")
        if execution is not None:
            lines.append("")
            lines.append("## Execution Summary")
            for item in execution["execution"]:
                status = "ok" if item["returncode"] == 0 else "failed"
                lines.append(f"- {item['name']}: {status}")
        lines.append("")
        lines.append("## Resume Points")
        lines.append("- Use the generated metrics files in results/ for model comparison.")
        lines.append("- Use the agent plan as a template for a real LLM-orchestrated workflow.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path


def _profile_entry() -> None:
    """Small helper used by the plan to profile the dataset."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=Path, default=None)
    args, _ = parser.parse_known_args()
    dataset = args.dataset or Path(__file__).resolve().parent / ".." / "lab-ids-anta-main" / "Dataset" / "normalized_data_2017.csv"
    agent = HybridIDSAgent(Path(__file__).resolve().parent, dataset, Path(__file__).resolve().parent / "results" / "agent")
    profile = agent.profile_dataset()
    print(json.dumps(profile, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_dataset = repo_root.parent / "lab-ids-anta-main" / "Dataset" / "normalized_data_2017.csv"
    default_output = repo_root / "results" / "agent"

    parser = argparse.ArgumentParser(description="Minimal agent-style orchestrator for Hybrid Scheme 1")
    parser.add_argument("--task", default="评估数据集鲁棒性并生成报告", help="Natural-language task description")
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="Dataset path")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory for agent outputs")
    parser.add_argument(
        "--feature-selection",
        choices=["forward", "backward", "correlation", "importance", "compare"],
        default=None,
        help="Force a feature-selection strategy",
    )
    parser.add_argument(
        "--adversarial-method",
        choices=["none", "fgsm", "owc-sawn", "both"],
        default=None,
        help="Force an adversarial training strategy",
    )
    parser.add_argument("--max-features", type=int, default=20, help="Maximum features to keep for selection")
    parser.add_argument("--execute", action="store_true", help="Execute the planned tool calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = HybridIDSAgent(Path(__file__).resolve().parent, args.dataset, args.output_dir)
    plan = agent.build_plan(
        task=args.task,
        feature_selection=args.feature_selection,
        adversarial_method=args.adversarial_method,
        max_features=args.max_features,
    )

    plan_path = agent.output_dir / "agent_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved plan to {plan_path}")

    if args.execute:
        execution = agent.execute(plan)
        execution_path = agent.output_dir / "agent_execution.json"
        execution_path.write_text(json.dumps(execution, indent=2, ensure_ascii=False), encoding="utf-8")
        report_path = agent.write_report(plan, execution)
        print(f"Saved execution log to {execution_path}")
        print(f"Saved report to {report_path}")
    else:
        report_path = agent.write_report(plan)
        print(f"Saved report to {report_path}")
        print("Run with --execute to actually invoke the pipeline tools.")


if __name__ == "__main__":
    main()