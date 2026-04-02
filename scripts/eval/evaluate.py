"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: scripts/eval/evaluate.py
Description: Comprehensive automatic evaluation comparing baseline Tamil-LLaMA with TanglishBridge.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from statistics import mean
from typing import Dict, List

import torch
from huggingface_hub import login
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tanglishbridge.detector import ScriptDetector
from tanglishbridge.pipeline import TanglishBridgePipeline


class BaselineTamilLLaMA:
    """
    Loads Tamil-LLaMA directly without TanglishBridge preprocessing.
    """

    def __init__(
        self,
        model_name: str = "abhinand/tamil-llama-7b-instruct-v0.1",
        use_4bit: bool = True,
        device: str = "auto",
    ) -> None:
        """
        Initialize the baseline model loader.

        Args:
            model_name: Hugging Face model identifier.
            use_4bit: Whether to prefer 4-bit quantized loading on GPU.
            device: ``auto``, ``cuda``, or ``cpu``.
        """
        try:
            print("[BaselineTamilLLaMA] Initializing baseline model...")
            self.model_name = model_name
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = None
            self.tokenizer = None
            self.model_available = False

            hf_token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
            if hf_token and hf_token != "YOUR_HF_TOKEN_HERE":
                login(token=hf_token, add_to_git_credential=False)

            model_kwargs: Dict[str, object] = {"trust_remote_code": True}
            if use_4bit and self.device == "cuda":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["device_map"] = "auto"
                model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                if self.device == "cuda":
                    model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.device == "cpu":
                self.model.to(self.device)
            self.model.eval()
            self.model_available = True
        except Exception as exc:
            logger.exception("Failed to initialize baseline Tamil-LLaMA: %s", exc)
            self.model_available = False

    def generate(self, text: str, max_new_tokens: int = 200) -> str:
        """
        Generate a raw baseline response without preprocessing.

        Args:
            text: Input query.
            max_new_tokens: Maximum new tokens.

        Returns:
            Decoded baseline response text.
        """
        try:
            print("[BaselineTamilLLaMA] Generating baseline response...")
            if not self.model_available:
                if "name" in text.lower():
                    return "My name is Tamil-LLaMA."
                if "தமிழ்" in text:
                    return "தமிழ் ஒரு மொழி."
                return "Ungal input received. Konjam more detail kudunga."

            prompt = f"### Instruction:\n{text}\n\n### Response:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            decoded = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            if not decoded:
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            return decoded
        except Exception as exc:
            logger.exception("Baseline generation failed: %s", exc)
            return "Baseline generation failed."


class TanglishBridgeEvaluator:
    """
    Runs automatic evaluation and saves conference-style reports.
    """

    def __init__(self) -> None:
        """
        Initialize paths, detector, and model wrappers.
        """
        try:
            print("[TanglishBridgeEvaluator] Initializing evaluator...")
            self.project_root = PROJECT_ROOT
            self.detector = ScriptDetector()
            self.pipeline = TanglishBridgePipeline()
            self.baseline = BaselineTamilLLaMA()
            self.test_inputs_path = os.path.join(self.project_root, "data", "processed", "test_inputs.json")
            self.metrics_path = os.path.join(self.project_root, "scripts", "eval", "metrics.json")
            self.report_path = os.path.join(self.project_root, "scripts", "eval", "evaluation_results.txt")
        except Exception as exc:
            logger.exception("Failed to initialize evaluator: %s", exc)

    def _load_test_cases(self) -> List[Dict[str, object]]:
        """
        Load curated evaluation cases from the processed data folder.

        Returns:
            List of evaluation case dictionaries.
        """
        try:
            print("[TanglishBridgeEvaluator] Loading curated test cases...")
            with open(self.test_inputs_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.exception("Failed to load test cases: %s", exc)
            return []

    def _score_script_handling(self, response: str, expected_script: str) -> float:
        """
        Score whether a response matches the desired script behavior.

        Args:
            response: Model response text.
            expected_script: Expected response style.

        Returns:
            Binary correctness score represented as ``0.0`` or ``1.0``.
        """
        try:
            print("[TanglishBridgeEvaluator] Scoring script handling...")
            stats = self.detector.get_text_stats(response)
            response_script = str(stats["script_type"])
            tamil_words = int(stats["tamil_words"])
            english_words = int(stats["english_words"])
            cmi_score = float(stats["cmi_score"])
            has_romanized = bool(stats["has_romanized"])

            if expected_script == "tamil":
                return 1.0 if tamil_words >= max(1, english_words) else 0.0
            if expected_script == "english":
                return 1.0 if english_words >= max(1, tamil_words) else 0.0
            if expected_script == "romanized":
                romanized_overlap = any(
                    token.lower().strip(".,?!") in self.detector.ROMANIZED_TAMIL_WORDS for token in response.split()
                )
                return 1.0 if response_script in {"romanized", "tanglish"} or has_romanized or romanized_overlap else 0.0
            if expected_script == "tanglish":
                return 1.0 if english_words > 0 and tamil_words > 0 and 0.1 <= cmi_score <= 0.8 else 0.0
            if expected_script == "mixed":
                return 1.0 if cmi_score >= 0.2 and english_words > 0 and tamil_words > 0 else 0.0
            return 0.0
        except Exception as exc:
            logger.exception("Failed to score script handling: %s", exc)
            return 0.0

    def _score_coherence(self, response: str, keywords: List[str]) -> float:
        """
        Score response coherence using keyword overlap heuristics.

        Args:
            response: Model response text.
            keywords: Expected relevant keywords.

        Returns:
            Score in the range ``[0, 1]``.
        """
        try:
            print("[TanglishBridgeEvaluator] Scoring response coherence...")
            lowered = response.lower()
            hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
            return round(hits / max(1, len(keywords)), 3)
        except Exception as exc:
            logger.exception("Failed to score coherence: %s", exc)
            return 0.0

    def _score_bleu(self, response: str, reference: str) -> float:
        """
        Compute sentence-level BLEU between response and reference.

        Args:
            response: Generated response.
            reference: Reference response.

        Returns:
            BLEU score in the range ``[0, 1]``.
        """
        try:
            print("[TanglishBridgeEvaluator] Computing BLEU score...")
            smoothing = SmoothingFunction().method1
            return round(
                sentence_bleu([reference.split()], response.split(), smoothing_function=smoothing),
                3,
            )
        except Exception as exc:
            logger.exception("Failed to compute BLEU score: %s", exc)
            return 0.0

    def _quality_summary(self, response: str) -> Dict[str, object]:
        """
        Produce simple response quality statistics.

        Args:
            response: Generated response text.

        Returns:
            Dictionary containing word count and completeness heuristics.
        """
        try:
            print("[TanglishBridgeEvaluator] Building quality summary...")
            words = response.split()
            length = len(words)
            return {
                "word_count": length,
                "completeness": bool(response.strip()) and response.strip()[-1] in ".!?",
                "good_length": 10 <= length <= 50,
                "too_short": length < 5,
                "too_long": length > 100,
            }
        except Exception as exc:
            logger.exception("Failed to build quality summary: %s", exc)
            return {"word_count": 0, "completeness": False, "good_length": False, "too_short": True, "too_long": False}

    def run(self) -> Dict[str, object]:
        """
        Execute evaluation, print formatted tables, and save outputs.

        Returns:
            Metrics dictionary summarizing baseline vs TanglishBridge performance.
        """
        try:
            print("[TanglishBridgeEvaluator] Running full evaluation...")
            cases = self._load_test_cases()
            results: List[Dict[str, object]] = []

            for case in cases:
                input_text = str(case["input"])
                expected_script = str(case["script_type"])
                keywords = list(case["keywords"])
                reference = str(case["reference_response"])

                baseline_response = self.baseline.generate(input_text)
                ours_result = self.pipeline.generate(input_text)
                ours_response = str(ours_result["final_response"])

                input_cmi = float(self.detector.calculate_cmi(input_text))
                baseline_cmi = float(self.detector.calculate_cmi(baseline_response))
                ours_cmi = float(self.detector.calculate_cmi(ours_response))

                results.append(
                    {
                        "id": case["id"],
                        "input": input_text,
                        "expected_script": expected_script,
                        "detected_script": ours_result["detected_script"],
                        "input_cmi": input_cmi,
                        "baseline_response": baseline_response,
                        "ours_response": ours_response,
                        "normalization_log": ours_result["processing_log"],
                        "baseline_script_acc": self._score_script_handling(baseline_response, expected_script),
                        "ours_script_acc": self._score_script_handling(ours_response, expected_script),
                        "baseline_coherence": self._score_coherence(baseline_response, keywords),
                        "ours_coherence": self._score_coherence(ours_response, keywords),
                        "baseline_cmi_gap": abs(input_cmi - baseline_cmi),
                        "ours_cmi_gap": abs(input_cmi - ours_cmi),
                        "baseline_bleu": self._score_bleu(baseline_response, reference),
                        "ours_bleu": self._score_bleu(ours_response, reference),
                        "baseline_quality": self._quality_summary(baseline_response),
                        "ours_quality": self._quality_summary(ours_response),
                    }
                )

            metrics = {
                "baseline_model_available": self.baseline.model_available,
                "pipeline_model_available": self.pipeline.model_available,
                "script_acc_baseline": round(mean(item["baseline_script_acc"] for item in results) * 100, 2),
                "script_acc_ours": round(mean(item["ours_script_acc"] for item in results) * 100, 2),
                "coherence_baseline": round(mean(item["baseline_coherence"] for item in results), 3),
                "coherence_ours": round(mean(item["ours_coherence"] for item in results), 3),
                "cmi_preservation_baseline": round(mean(item["baseline_cmi_gap"] for item in results), 3),
                "cmi_preservation_ours": round(mean(item["ours_cmi_gap"] for item in results), 3),
                "bleu_baseline": round(mean(item["baseline_bleu"] for item in results), 3),
                "bleu_ours": round(mean(item["ours_bleu"] for item in results), 3),
                "avg_response_len_baseline": round(mean(item["baseline_quality"]["word_count"] for item in results), 2),
                "avg_response_len_ours": round(mean(item["ours_quality"]["word_count"] for item in results), 2),
                "good_length_baseline": round(mean(1.0 if item["baseline_quality"]["good_length"] else 0.0 for item in results) * 100, 2),
                "good_length_ours": round(mean(1.0 if item["ours_quality"]["good_length"] else 0.0 for item in results) * 100, 2),
                "results": results,
            }

            delta_script = round(metrics["script_acc_ours"] - metrics["script_acc_baseline"], 2)
            delta_coherence = round(metrics["coherence_ours"] - metrics["coherence_baseline"], 3)
            delta_cmi = round(metrics["cmi_preservation_ours"] - metrics["cmi_preservation_baseline"], 3)
            delta_len = round(metrics["avg_response_len_ours"] - metrics["avg_response_len_baseline"], 2)
            delta_good_len = round(metrics["good_length_ours"] - metrics["good_length_baseline"], 2)

            table = [
                "╔══════════════════════════════════════════════════════════╗",
                "║         TANGLISHBRIDGE EVALUATION RESULTS               ║",
                "╠══════════════════════════════════════════════════════════╣",
                "║ Metric              │ Baseline │ TanglishBridge │ Δ      ║",
                "╠══════════════════════════════════════════════════════════╣",
                f"║ Script Acc (%)      │ {metrics['script_acc_baseline']:7.2f} │ {metrics['script_acc_ours']:13.2f} │ {delta_script:+6.2f} ║",
                f"║ Coherence Score     │ {metrics['coherence_baseline']:7.3f} │ {metrics['coherence_ours']:13.3f} │ {delta_coherence:+6.3f} ║",
                f"║ CMI Preservation    │ {metrics['cmi_preservation_baseline']:7.3f} │ {metrics['cmi_preservation_ours']:13.3f} │ {delta_cmi:+6.3f} ║",
                f"║ Avg Response Len    │ {metrics['avg_response_len_baseline']:7.2f} │ {metrics['avg_response_len_ours']:13.2f} │ {delta_len:+6.2f} ║",
                f"║ Good Length %       │ {metrics['good_length_baseline']:7.2f} │ {metrics['good_length_ours']:13.2f} │ {delta_good_len:+6.2f} ║",
                "╚══════════════════════════════════════════════════════════╝",
            ]

            qualitative_rows = []
            if not self.pipeline.model_available or not self.baseline.model_available:
                qualitative_rows.extend(
                    [
                        "",
                        "NOTE: The Tamil-LLaMA model did not load in this environment.",
                        "The outputs below were produced with fallback heuristic responses, so these metrics are not true model evaluation scores.",
                        "-" * 80,
                    ]
                )

            qualitative_rows.extend(["", "QUALITATIVE COMPARISON", "-" * 80])
            for item in results:
                qualitative_rows.append(f"Input [{item['id']}]: {item['input']}")
                qualitative_rows.append(f"Baseline: {item['baseline_response']}")
                qualitative_rows.append(f"TanglishBridge: {item['ours_response']}")
                qualitative_rows.append("-" * 80)

            report_text = "\n".join(table + qualitative_rows)
            print(report_text)

            os.makedirs(os.path.join(self.project_root, "scripts", "eval"), exist_ok=True)
            with open(self.metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, ensure_ascii=False, indent=2)
            with open(self.report_path, "w", encoding="utf-8") as handle:
                handle.write(report_text)

            return metrics
        except Exception as exc:
            logger.exception("Evaluation run failed: %s", exc)
            return {"error": str(exc)}


def main() -> None:
    """
    Script entrypoint for automatic TanglishBridge evaluation.
    """
    try:
        print("[evaluate.py] Starting evaluation script...")
        evaluator = TanglishBridgeEvaluator()
        evaluator.run()
    except Exception as exc:
        logger.exception("evaluate.py failed: %s", exc)


if __name__ == "__main__":
    main()
