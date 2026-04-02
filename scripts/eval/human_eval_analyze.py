"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: scripts/eval/human_eval_analyze.py
Description: Analyzes completed human evaluation sheets and computes summary metrics.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

import pandas as pd
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


class HumanEvalAnalyzer:
    """
    Aggregates metric means, preferences, and annotator agreement from a filled CSV sheet.
    """

    def __init__(self) -> None:
        """
        Initialize input and output paths.
        """
        try:
            print("[HumanEvalAnalyzer] Initializing analyzer...")
            self.input_path = os.path.join(PROJECT_ROOT, "scripts", "eval", "human_eval_sheet.csv")
            self.output_path = os.path.join(PROJECT_ROOT, "scripts", "eval", "human_eval_results.json")
        except Exception as exc:
            logger.exception("Failed to initialize HumanEvalAnalyzer: %s", exc)

    def _compute_metric_kappa(self, df: pd.DataFrame, metric_name: str) -> str:
        """
        Compute Cohen's Kappa across annotators for a derived metric winner label.

        Args:
            df: Human evaluation dataframe.
            metric_name: Metric prefix such as ``Fluency``.

        Returns:
            Kappa formatted as a string or ``N/A`` when insufficient annotators exist.
        """
        try:
            print("[HumanEvalAnalyzer] Computing Cohen's Kappa...")
            if "Annotator_ID" not in df.columns:
                return "N/A"

            working = df.copy()
            winner_column = f"{metric_name}_Winner"
            baseline_col = f"{metric_name}_Baseline(1-5)"
            ours_col = f"{metric_name}_Ours(1-5)"
            working[winner_column] = working.apply(
                lambda row: "Ours"
                if float(row[ours_col]) > float(row[baseline_col])
                else ("Baseline" if float(row[baseline_col]) > float(row[ours_col]) else "Equal"),
                axis=1,
            )

            annotators = sorted(working["Annotator_ID"].dropna().unique().tolist())
            if len(annotators) < 2:
                return "N/A"

            pivot = working.pivot_table(index="ID", columns="Annotator_ID", values=winner_column, aggfunc="first")
            pivot = pivot.dropna()
            if pivot.shape[1] < 2 or pivot.empty:
                return "N/A"

            first_two = pivot.iloc[:, :2]
            return f"{cohen_kappa_score(first_two.iloc[:, 0], first_two.iloc[:, 1]):.2f}"
        except Exception as exc:
            logger.exception("Failed to compute metric kappa: %s", exc)
            return "N/A"

    def run(self) -> Dict[str, object]:
        """
        Analyze the filled human evaluation sheet and save a JSON summary.

        Returns:
            Dictionary with means, preferences, and kappa values.
        """
        try:
            print("[HumanEvalAnalyzer] Running human evaluation analysis...")
            df = pd.read_csv(self.input_path)

            results = {
                "Fluency": {
                    "baseline": round(float(df["Fluency_Baseline(1-5)"].mean()), 2),
                    "ours": round(float(df["Fluency_Ours(1-5)"].mean()), 2),
                    "kappa": self._compute_metric_kappa(df, "Fluency"),
                },
                "Naturalness": {
                    "baseline": round(float(df["Naturalness_Baseline(1-5)"].mean()), 2),
                    "ours": round(float(df["Naturalness_Ours(1-5)"].mean()), 2),
                    "kappa": self._compute_metric_kappa(df, "Naturalness"),
                },
                "Correctness": {
                    "baseline": round(float(df["Correctness_Baseline(1-5)"].mean()), 2),
                    "ours": round(float(df["Correctness_Ours(1-5)"].mean()), 2),
                    "kappa": self._compute_metric_kappa(df, "Correctness"),
                },
                "Preference": {
                    "baseline_percent": round(float((df["Preference(Baseline/Ours/Equal)"] == "Baseline").mean() * 100), 2),
                    "ours_percent": round(float((df["Preference(Baseline/Ours/Equal)"] == "Ours").mean() * 100), 2),
                    "equal_percent": round(float((df["Preference(Baseline/Ours/Equal)"] == "Equal").mean() * 100), 2),
                },
            }

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)

            print("┌─────────────────┬──────────┬────────┬───────┐")
            print("│ Metric          │ Baseline │  Ours  │ Kappa │")
            print("├─────────────────┼──────────┼────────┼───────┤")
            print(
                f"│ Fluency         │ {results['Fluency']['baseline']:>8.2f} │ {results['Fluency']['ours']:>6.2f} │ {results['Fluency']['kappa']:>5} │"
            )
            print(
                f"│ Naturalness     │ {results['Naturalness']['baseline']:>8.2f} │ {results['Naturalness']['ours']:>6.2f} │ {results['Naturalness']['kappa']:>5} │"
            )
            print(
                f"│ Correctness     │ {results['Correctness']['baseline']:>8.2f} │ {results['Correctness']['ours']:>6.2f} │ {results['Correctness']['kappa']:>5} │"
            )
            print(
                f"│ Preference (%)  │ {results['Preference']['baseline_percent']:>7.2f}% │ {results['Preference']['ours_percent']:>6.2f}% │ {'-':>5} │"
            )
            print("└─────────────────┴──────────┴────────┴───────┘")

            return results
        except Exception as exc:
            logger.exception("Human evaluation analysis failed: %s", exc)
            return {"error": str(exc)}


def main() -> None:
    """
    Script entrypoint for human evaluation analysis.
    """
    try:
        print("[human_eval_analyze.py] Starting human evaluation analysis...")
        analyzer = HumanEvalAnalyzer()
        analyzer.run()
    except Exception as exc:
        logger.exception("human_eval_analyze.py failed: %s", exc)


if __name__ == "__main__":
    main()
