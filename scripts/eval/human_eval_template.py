"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: scripts/eval/human_eval_template.py
Description: Generates a 50-case human evaluation sheet with baseline and TanglishBridge responses.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from evaluate import BaselineTamilLLaMA
from tanglishbridge.pipeline import TanglishBridgePipeline


class HumanEvalSheetBuilder:
    """
    Builds the CSV sheet used for human annotation.
    """

    def __init__(self) -> None:
        """
        Initialize output paths and model wrappers.
        """
        try:
            print("[HumanEvalSheetBuilder] Initializing sheet builder...")
            self.output_path = os.path.join(PROJECT_ROOT, "scripts", "eval", "human_eval_sheet.csv")
            self.pipeline = TanglishBridgePipeline()
            self.baseline = BaselineTamilLLaMA()
        except Exception as exc:
            logger.exception("Failed to initialize HumanEvalSheetBuilder: %s", exc)

    def _build_cases(self) -> List[Dict[str, str]]:
        """
        Build the curated 50-item human evaluation test set.

        Returns:
            List of dictionaries with ``id``, ``script_type``, and ``input_text`` fields.
        """
        try:
            print("[HumanEvalSheetBuilder] Building curated human evaluation cases...")
            tanglish = [
                "bro saptiya?",
                "naan office ku varen",
                "enna da pannure?",
                "tired ah iruku da",
                "office eppo open aagum?",
                "meeting mudinju call pannu",
                "coffee sapdala na headache varum",
                "late ah varren, wait pannunga",
                "traffic romba iruku bro",
                "weekend plan enna da?",
                "phone charge illa, msg pannunga",
                "project deadline nala stress ah iruken",
                "laptop hang aaguthu da",
                "rain nala office leave iruka?",
                "super da, presentation nalla pochu",
            ]
            romanized = [
                "vanakkam, nee eppadi irukka?",
                "nandri sollu da",
                "saapta? illa poi saptu vaa",
                "naan nalla irukken",
                "neenga enga irukkeenga?",
                "intha padam romba nalla iruku",
                "konjam help pannunga",
                "enna aachu inniku?",
                "naalaiku varalama?",
                "veetla ellarum nalla iruka?",
            ]
            tamil = [
                "தமிழ் என்றால் என்ன?",
                "உன் பெயர் என்ன?",
                "நீ இன்று எப்படி இருக்கிறாய்?",
                "மழை பெய்கிறதா?",
                "இன்று அலுவலகம் திறந்திருக்கும்?",
                "தமிழ் மொழியின் சிறப்பு என்ன?",
                "நன்றி சொல்ல எப்படி?",
                "இந்த வார இறுதியில் என்ன செய்யலாம்?",
                "நான் ஓய்வு எடுக்க வேண்டுமா?",
                "இந்த தேர்வு முடிவு எப்போது வரும்?",
                "நல்ல நண்பன் என்றால் யார்?",
                "புதிய திட்டத்தை எப்படி தொடங்குவது?",
                "இன்று போக்குவரத்து நெரிசல் அதிகமா?",
                "நான் இன்று தாமதமாக வருவேன்.",
                "உதவி தேவைப்பட்டால் என்ன செய்ய வேண்டும்?",
            ]
            english = [
                "what is your name?",
                "how are you today?",
                "can you help me with my project?",
                "when will the office open?",
                "is it raining outside?",
                "what is the weekend plan?",
                "should I take some rest now?",
                "did the exam result come?",
                "please send me the phone number",
                "why is the traffic so bad today?",
            ]

            cases: List[Dict[str, str]] = []
            index = 1
            for script_type, items in [
                ("tanglish", tanglish),
                ("romanized", romanized),
                ("tamil", tamil),
                ("english", english),
            ]:
                for item in items:
                    cases.append(
                        {
                            "id": f"HE{index:03d}",
                            "script_type": script_type,
                            "input_text": item,
                        }
                    )
                    index += 1
            return cases
        except Exception as exc:
            logger.exception("Failed to build human evaluation cases: %s", exc)
            return []

    def run(self) -> str:
        """
        Generate the filled annotation CSV and print evaluator instructions.

        Returns:
            Output CSV path.
        """
        try:
            print("[HumanEvalSheetBuilder] Generating human evaluation CSV...")
            rows = []
            for case in self._build_cases():
                baseline_response = self.baseline.generate(case["input_text"])
                pipeline_response = self.pipeline.generate(case["input_text"])["final_response"]
                rows.append(
                    {
                        "ID": case["id"],
                        "Script_Type": case["script_type"],
                        "Input_Text": case["input_text"],
                        "Baseline_Response": baseline_response,
                        "TanglishBridge_Response": pipeline_response,
                        "Fluency_Baseline(1-5)": "",
                        "Fluency_Ours(1-5)": "",
                        "Naturalness_Baseline(1-5)": "",
                        "Naturalness_Ours(1-5)": "",
                        "Correctness_Baseline(1-5)": "",
                        "Correctness_Ours(1-5)": "",
                        "Preference(Baseline/Ours/Equal)": "",
                        "Notes": "",
                    }
                )

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            print("Human evaluation instructions:")
            print("1. Rate Fluency, Naturalness, and Correctness independently on a 1-5 scale.")
            print("2. Use Preference to mark Baseline, Ours, or Equal for the overall better answer.")
            print("3. Add short Notes for translation errors, awkward phrasing, or code-mixing issues.")
            print("4. If you have multiple annotators, add an Annotator_ID column before merging sheets.")
            print(f"5. Saved annotation sheet to: {self.output_path}")
            return self.output_path
        except Exception as exc:
            logger.exception("Failed to generate human evaluation sheet: %s", exc)
            return self.output_path


def main() -> None:
    """
    Script entrypoint for generating the human evaluation CSV.
    """
    try:
        print("[human_eval_template.py] Starting human evaluation sheet generation...")
        builder = HumanEvalSheetBuilder()
        builder.run()
    except Exception as exc:
        logger.exception("human_eval_template.py failed: %s", exc)


if __name__ == "__main__":
    main()
