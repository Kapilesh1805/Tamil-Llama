"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: scripts/eval/cmi_analysis.py
Description: Dataset-level CMI analysis, categorization, and plot generation for TanglishBridge.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List

import matplotlib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tanglishbridge.detector import ScriptDetector


class CMIAnalyzer:
    """
    Computes code-mixing statistics over the available raw dataset.
    """

    def __init__(self) -> None:
        """
        Initialize dataset paths and detector dependencies.
        """
        try:
            print("[CMIAnalyzer] Initializing CMI analyzer...")
            self.project_root = PROJECT_ROOT
            self.detector = ScriptDetector()
            self.raw_dir = os.path.join(self.project_root, "data", "raw")
            self.processed_dir = os.path.join(self.project_root, "data", "processed")
            self.analysis_path = os.path.join(self.processed_dir, "cmi_analysis.json")
            self.stats_path = os.path.join(self.processed_dir, "cmi_stats.json")
        except Exception as exc:
            logger.exception("Failed to initialize CMIAnalyzer: %s", exc)

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load the best available raw dataset and normalize its text column.

        Returns:
            DataFrame with a ``text`` column for CMI scoring.
        """
        try:
            print("[CMIAnalyzer] Loading raw dataset...")
            preferred = os.path.join(self.raw_dir, "tamil_sentiment_full.csv")
            alternate = os.path.join(self.raw_dir, "tamil_sentiment_full_train.csv")
            tamil_csv = os.path.join(self.raw_dir, "Tamil.csv")

            if os.path.exists(preferred):
                df = pd.read_csv(preferred)
            elif os.path.exists(alternate):
                df = pd.read_csv(alternate, sep="\t", names=["text", "label"], engine="python")
            elif os.path.exists(tamil_csv):
                df = pd.read_csv(tamil_csv)
            else:
                raise FileNotFoundError("No compatible raw dataset found in data/raw.")

            if "Transcript" in df.columns:
                df["text"] = df["Transcript"].astype(str)
            elif "text" not in df.columns:
                first_column = df.columns[0]
                df["text"] = df[first_column].astype(str)
            else:
                df["text"] = df["text"].astype(str)

            df = df[df["text"].str.strip().ne("")]
            return df
        except Exception as exc:
            logger.exception("Failed to load dataset for CMI analysis: %s", exc)
            return pd.DataFrame(columns=["text"])

    def run(self) -> Dict[str, object]:
        """
        Analyze CMI distribution, save plots, and persist summary JSON files.

        Returns:
            Dictionary with category counts, percentages, and averages.
        """
        try:
            print("[CMIAnalyzer] Running dataset-level CMI analysis...")
            os.makedirs(self.processed_dir, exist_ok=True)
            df = self._load_dataset()
            if df.empty:
                raise ValueError("Dataset is empty; cannot compute CMI statistics.")

            df["cmi"] = df["text"].apply(self.detector.calculate_cmi)
            df["category"] = pd.cut(
                df["cmi"],
                bins=[-0.001, 0.3, 0.6, 1.0],
                labels=["Low mixing", "Medium mix", "High mixing"],
            )

            summary_rows: List[Dict[str, object]] = []
            for label in ["Low mixing", "Medium mix", "High mixing"]:
                subset = df[df["category"] == label]
                count = int(len(subset))
                percent = round((count / len(df)) * 100, 2)
                avg_cmi = round(float(subset["cmi"].mean()), 3) if count else 0.0
                summary_rows.append({"category": label, "count": count, "percent": percent, "avg_cmi": avg_cmi})

            overall_avg = round(float(df["cmi"].mean()), 3)

            histogram_path = os.path.join(self.processed_dir, "cmi_histogram.png")
            piechart_path = os.path.join(self.processed_dir, "cmi_piechart.png")

            plt.figure(figsize=(8, 5))
            plt.hist(df["cmi"], bins=20, color="#2f6b5f", edgecolor="#f7efe1")
            plt.title("TanglishBridge CMI Distribution")
            plt.xlabel("CMI Score")
            plt.ylabel("Sentence Count")
            plt.tight_layout()
            plt.savefig(histogram_path, dpi=200)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.pie(
                [row["count"] for row in summary_rows],
                labels=[row["category"] for row in summary_rows],
                autopct="%1.1f%%",
                colors=["#d9ead3", "#ffd966", "#f4cccc"],
                startangle=140,
            )
            plt.title("TanglishBridge CMI Category Split")
            plt.tight_layout()
            plt.savefig(piechart_path, dpi=200)
            plt.close()

            results = {
                "dataset_size": int(len(df)),
                "overall_average_cmi": overall_avg,
                "categories": summary_rows,
                "histogram_path": histogram_path,
                "piechart_path": piechart_path,
            }

            with open(self.analysis_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
            with open(self.stats_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)

            print("┌──────────────┬───────┬────────┬─────────┐")
            print("│ Category     │ Count │   %    │ Avg CMI │")
            print("├──────────────┼───────┼────────┼─────────┤")
            for row in summary_rows:
                print(
                    f"│ {row['category']:<12} │ {row['count']:>5} │ {row['percent']:>6.2f}% │ {row['avg_cmi']:>7.3f} │"
                )
            print(f"│ {'Total':<12} │ {len(df):>5} │ {100.00:>6.2f}% │ {overall_avg:>7.3f} │")
            print("└──────────────┴───────┴────────┴─────────┘")

            return results
        except Exception as exc:
            logger.exception("CMI analysis failed: %s", exc)
            return {"error": str(exc)}


def main() -> None:
    """
    Script entrypoint for dataset-level CMI analysis.
    """
    try:
        print("[cmi_analysis.py] Starting CMI analysis...")
        analyzer = CMIAnalyzer()
        analyzer.run()
    except Exception as exc:
        logger.exception("cmi_analysis.py failed: %s", exc)


if __name__ == "__main__":
    main()
