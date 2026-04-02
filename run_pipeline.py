"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: run_pipeline.py
Description: Quick standalone script for testing TanglishBridge on a curated input set.
"""

from __future__ import annotations

import logging
from statistics import mean
from typing import Dict, List

from tanglishbridge.pipeline import TanglishBridgePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineQuickRunner:
    """
    Runs a compact TanglishBridge smoke test over 10 representative inputs.
    """

    def __init__(self) -> None:
        """
        Initialize the TanglishBridge pipeline and test set.
        """
        try:
            print("[PipelineQuickRunner] Initializing quick test runner...")
            self.pipeline = TanglishBridgePipeline()
            self.test_inputs = [
                "bro saptiya?",
                "vanakkam da",
                "தமிழ் என்றால் என்ன?",
                "what is your name?",
                "naan office ku varen, traffic romba iruku",
                "tired ah iruku da, rest edukanum",
                "exam eppo da?",
                "rain aaguthu, umbrella edutukitiya?",
                "weekend plan enna?",
                "super da, congrats!",
            ]
        except Exception as exc:
            logger.exception("Failed to initialize PipelineQuickRunner: %s", exc)
            self.test_inputs = []

    def run(self) -> List[Dict[str, object]]:
        """
        Run the pipeline over the curated test inputs and print a formatted summary.

        Returns:
            List of pipeline result dictionaries.
        """
        try:
            print("[PipelineQuickRunner] Running quick pipeline test...")
            results = self.pipeline.batch_generate(self.test_inputs)
            for result in results:
                print("┌─────────────────────────────────────────┐")
                print(f"│ Input: {result['input'][:34]:<34} │")
                print(f"│ Script: {result['detected_script'].upper():<9} | CMI: {result['cmi_score']:<11.3f} │")
                print(f"│ Normalized: {result['normalized_input'][:27]:<27} │")
                print(f"│ Response: {result['final_response'][:29]:<29} │")
                print("└─────────────────────────────────────────┘")

            stats = self.pipeline.get_pipeline_stats()
            average_response_length = mean(len(item["final_response"].split()) for item in results) if results else 0.0
            print(f"Total processed: {len(results)}")
            print(f"Script types detected: {stats['script_counts']}")
            print(f"Average CMI: {stats['average_cmi']:.3f}")
            print(f"Average response length: {average_response_length:.2f} words")
            return results
        except Exception as exc:
            logger.exception("Quick pipeline run failed: %s", exc)
            return []


def main() -> None:
    """
    Script entrypoint for the quick TanglishBridge smoke test.
    """
    try:
        print("[run_pipeline.py] Starting quick pipeline test...")
        runner = PipelineQuickRunner()
        runner.run()
    except Exception as exc:
        logger.exception("run_pipeline.py failed: %s", exc)


if __name__ == "__main__":
    main()
