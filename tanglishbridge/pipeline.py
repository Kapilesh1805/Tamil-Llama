"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: pipeline.py
Description: End-to-end TanglishBridge pipeline that wraps Tamil-LLaMA without fine-tuning.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tanglishbridge.detector import ScriptDetector
from tanglishbridge.normalizer import TanglishNormalizer
from tanglishbridge.postprocessor import ResponsePostProcessor
from tanglishbridge.transliterator import RomanizedTamilTransliterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TanglishBridgePipeline:
    """
    Wraps Tamil-LLaMA with detection, normalization, transliteration, and post-processing.
    """

    def __init__(
        self,
        model_name: str = "abhinand/tamil-llama-7b-instruct-v0.1",
        use_4bit: bool = True,
        device: str = "auto",
    ) -> None:
        """
        Initialize all TanglishBridge modules and try to load the base Tamil-LLaMA model.

        Args:
            model_name: Hugging Face model identifier.
            use_4bit: Whether to prefer 4-bit quantized loading on GPU.
            device: ``auto``, ``cuda``, or ``cpu``.
        """
        try:
            print("[TanglishBridgePipeline] Initializing TanglishBridge pipeline...")
            self.model_name = model_name
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.detector = ScriptDetector()
            self.normalizer = TanglishNormalizer()
            self.transliterator = RomanizedTamilTransliterator()
            self.postprocessor = ResponsePostProcessor()
            self.model = None
            self.tokenizer = None
            self.model_available = False
            self.model_unavailable_reason = ""
            self.stats = {
                "total_processed": 0,
                "script_counts": {"tamil": 0, "english": 0, "tanglish": 0, "romanized": 0, "mixed": 0},
                "total_cmi": 0.0,
                "total_response_words": 0,
                "model_name": model_name,
                "device": self.device,
            }
            self._load_model(use_4bit=use_4bit)
        except Exception as exc:
            logger.exception("Failed to initialize TanglishBridgePipeline: %s", exc)

    def _load_model(self, use_4bit: bool) -> None:
        """
        Load tokenizer and Tamil-LLaMA weights with device-aware configuration.

        Args:
            use_4bit: Whether to enable 4-bit quantization when supported.
        """
        try:
            print("[TanglishBridgePipeline] Loading Tamil-LLaMA model and tokenizer...")
            hf_token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
            if hf_token and hf_token != "YOUR_HF_TOKEN_HERE":
                login(token=hf_token, add_to_git_credential=False)
            else:
                logger.info("HF_TOKEN not provided; attempting anonymous Hugging Face access.")

            can_use_4bit = use_4bit and self.device == "cuda"
            model_kwargs: Dict[str, object] = {"trust_remote_code": True}

            if self.device == "cpu":
                try:
                    import psutil

                    memory = psutil.virtual_memory()
                    total_gb = round(memory.total / (1024**3), 2)
                    available_gb = round(memory.available / (1024**3), 2)
                    logger.info("CPU memory detected: total=%s GB, available=%s GB", total_gb, available_gb)
                    if "7b" in self.model_name.lower() and total_gb < 24:
                        self.model_unavailable_reason = (
                            f"Insufficient system RAM for loading {self.model_name} on CPU "
                            f"(total={total_gb} GB, available={available_gb} GB). "
                            "Use GPU/Colab, Ollama/GGUF, or a smaller local model."
                        )
                        logger.warning(self.model_unavailable_reason)
                        self.model_available = False
                        return
                except Exception as memory_exc:
                    logger.info("RAM pre-check skipped: %s", memory_exc)

            if can_use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
                model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                if self.device == "cuda":
                    model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.device == "cpu":
                self.model.to(self.device)
            self.model.eval()
            self.model_available = True
            self.model_unavailable_reason = ""
            logger.info("Tamil-LLaMA loaded successfully on %s.", self.device)
        except Exception as exc:
            self.model_available = False
            self.model_unavailable_reason = str(exc)
            logger.exception("Model loading failed. Falling back to heuristic responses: %s", exc)

    def _build_prompt(self, input_text: str, model_input: str, script_type: str) -> str:
        """
        Build a style-aware prompt that keeps the user's language register intact.

        Args:
            input_text: Original user message.
            model_input: Normalized or transliterated hint for the model.
            script_type: Detected script category.

        Returns:
            An instruction-formatted prompt string.
        """
        try:
            print("[TanglishBridgePipeline] Building generation prompt...")
            style_instructions = {
                "tanglish": (
                    "Reply to the following casual Tamil message in one short natural conversational Tamil reply. "
                    "Answer the message directly. Do not explain it. Do not translate it. Do not analyze it. "
                    "Do not say you are an AI unless asked."
                ),
                "romanized": (
                    "Reply to the following Tamil message in one short natural conversational Tamil reply. "
                    "Answer the message directly. Do not explain it. Do not translate it. Do not analyze it. "
                    "Do not say you are an AI unless asked."
                ),
                "tamil": (
                    "கீழே உள்ள செய்திக்கு ஒரு குறுகிய இயல்பான தமிழ் பதில் அளிக்கவும். "
                    "செய்தியை விளக்காதீர்கள். மொழிபெயர்க்காதீர்கள். நேராக பதிலளிக்கவும்."
                ),
                "english": (
                    "Reply to the following message in one short natural English reply. "
                    "Answer directly. Do not explain or translate the message."
                ),
                "mixed": (
                    "Reply to the following message in one short natural conversational Tamil reply. "
                    "Keep English technical words only when necessary. Answer directly. "
                    "Do not explain or translate the message."
                ),
            }
            instruction = style_instructions.get(script_type, style_instructions["mixed"])
            message_block = model_input if script_type in {"tanglish", "romanized", "mixed"} else input_text
            return f"### Instruction:\n{instruction}\n\n{message_block}\n\n### Response:\n"
        except Exception as exc:
            logger.exception("Error while building prompt: %s", exc)
            return f"### Instruction:\nAnswer naturally.\n\nUser message: {input_text}\n\n### Response:\n"

    def _is_explanatory_response(self, response: str) -> bool:
        """
        Detect whether a generated response is explaining the input instead of replying.

        Args:
            response: Raw generated model text.

        Returns:
            ``True`` when the response appears translation-like or explanatory.
        """
        try:
            print("[TanglishBridgePipeline] Checking whether response is explanatory...")
            lowered = response.lower()
            markers = [
                "பொருள்",
                "அதாவது",
                "என்பது",
                "ஆங்கிலத்தில்",
                "தமிழில்",
                "means",
                "in english",
                "in tamil",
                "translation",
            ]
            return any(marker in lowered for marker in markers)
        except Exception as exc:
            logger.exception("Error while checking explanatory response: %s", exc)
            return False

    def _build_reply_only_prompt(self, model_input: str, script_type: str) -> str:
        """
        Build a stricter retry prompt that asks only for a direct reply.

        Args:
            model_input: Clean model-facing text.
            script_type: Detected script category.

        Returns:
            A stricter instruction-formatted prompt string.
        """
        try:
            print("[TanglishBridgePipeline] Building reply-only prompt...")
            if script_type == "english":
                instruction = (
                    "Reply directly to this message in one short English sentence. "
                    "Do not explain, translate, or define anything."
                )
            else:
                instruction = (
                    "இந்த செய்திக்கு நேராக ஒரு குறுகிய இயல்பான பதில் சொல்லவும். "
                    "விளக்காதீர்கள். மொழிபெயர்க்காதீர்கள். செய்தியை பற்றி பேசாதீர்கள்."
                )
            return f"### Instruction:\n{instruction}\n\n{model_input}\n\n### Response:\n"
        except Exception as exc:
            logger.exception("Error while building reply-only prompt: %s", exc)
            return f"### Instruction:\nReply directly.\n\n{model_input}\n\n### Response:\n"

    def _prepare_model_text(self, text: str, script_type: str) -> str:
        """
        Convert mixed colloquial input into a cleaner model-facing text.

        Args:
            text: Normalized or transliterated text.
            script_type: Detected script category.

        Returns:
            A cleaner Tamil-first text for model prompting.
        """
        try:
            print("[TanglishBridgePipeline] Preparing model-facing text...")
            if script_type not in {"tanglish", "romanized", "mixed"}:
                return text

            filler_words = {
                "bro",
                "da",
                "di",
                "dei",
                "macha",
                "machi",
                "anna",
                "akka",
                "pa",
                "ma",
            }
            tokens = text.split()
            if len(tokens) > 1:
                tokens = [token for token in tokens if token.lower().strip(".,!?") not in filler_words]

            cleaned = " ".join(tokens).strip()
            cleaned = cleaned or text
            if script_type in {"tanglish", "romanized"} and not any("\u0B80" <= ch <= "\u0BFF" for ch in cleaned):
                transliterated = self.transliterator.smart_transliterate(cleaned)
                if transliterated:
                    cleaned = transliterated
            return cleaned
        except Exception as exc:
            logger.exception("Error while preparing model text: %s", exc)
            return text

    def generate(self, input_text: str, max_new_tokens: int = 200, fast_mode: bool = False) -> Dict[str, object]:
        """
        Run the full TanglishBridge pipeline for a single user input.

        Args:
            input_text: Raw user message.
            max_new_tokens: Maximum new tokens for generation.
            fast_mode: Whether to use a lower-latency generation configuration.

        Returns:
            A dictionary containing intermediate and final pipeline artifacts.
        """
        try:
            print("[TanglishBridgePipeline] Running end-to-end generation...")
            effective_max_new_tokens = max_new_tokens
            if self.device == "cpu":
                effective_max_new_tokens = min(max_new_tokens, 64)
            if fast_mode:
                effective_max_new_tokens = min(effective_max_new_tokens, 24 if self.device == "cpu" else 64)
            text_stats = self.detector.get_text_stats(input_text)
            script_type = str(text_stats["script_type"])
            cmi_score = float(text_stats["cmi_score"])
            processing_log: List[str] = [
                f"script detected: {script_type}",
                f"cmi score: {cmi_score}",
                f"max_new_tokens used: {effective_max_new_tokens}",
                f"fast_mode: {fast_mode}",
            ]

            normalized_input = input_text
            if script_type in {"tanglish", "romanized", "mixed"}:
                normalized_input = self.normalizer.normalize(input_text)
                processing_log.extend(self.normalizer.get_normalization_log(input_text))
            else:
                processing_log.append("normalization skipped for monolingual input")

            model_input = normalized_input
            if script_type in {"tanglish", "romanized", "mixed"}:
                transliterated_input = self.transliterator.smart_transliterate(normalized_input)
                if transliterated_input != normalized_input:
                    model_input = transliterated_input
                    processing_log.append(f"transliteration applied: {normalized_input} -> {model_input}")
                else:
                    processing_log.append("transliteration not required")
                prepared_input = self._prepare_model_text(model_input, script_type)
                if prepared_input != model_input:
                    processing_log.append(f"model text prepared: {model_input} -> {prepared_input}")
                    model_input = prepared_input
            else:
                processing_log.append("transliteration skipped for monolingual input")

            prompt = self._build_prompt(input_text=input_text, model_input=model_input, script_type=script_type)

            if self.model_available:
                raw_response = self._generate_with_model(
                    prompt=prompt,
                    max_new_tokens=effective_max_new_tokens,
                    fast_mode=fast_mode,
                )
                if script_type in {"tanglish", "romanized", "mixed", "tamil"} and self._is_explanatory_response(raw_response):
                    processing_log.append("explanatory response detected; retrying with reply-only prompt")
                    raw_response = self._generate_with_model(
                        prompt=self._build_reply_only_prompt(model_input=model_input, script_type=script_type),
                        max_new_tokens=effective_max_new_tokens,
                        fast_mode=True,
                    )
            else:
                raw_response = self._fallback_response(input_text=input_text, script_type=script_type)
                processing_log.append("heuristic fallback response used because the model is unavailable")

            response_style = script_type if script_type in {"tanglish", "romanized", "tamil", "english"} else "tanglish"
            final_response = self.postprocessor.process(raw_response, response_style)
            processing_log.append(f"post-processing style: {response_style}")

            self.stats["total_processed"] += 1
            self.stats["script_counts"][script_type] = self.stats["script_counts"].get(script_type, 0) + 1
            self.stats["total_cmi"] += cmi_score
            self.stats["total_response_words"] += len(final_response.split())

            return {
                "input": input_text,
                "detected_script": script_type,
                "cmi_score": cmi_score,
                "normalized_input": normalized_input,
                "model_input": model_input,
                "raw_response": raw_response,
                "final_response": final_response,
                "processing_log": processing_log,
            }
        except Exception as exc:
            logger.exception("Error during pipeline generation: %s", exc)
            return {
                "input": input_text,
                "detected_script": "mixed",
                "cmi_score": 0.0,
                "normalized_input": input_text,
                "model_input": input_text,
                "raw_response": "",
                "final_response": "I am here to help. Please try again with a little more detail.",
                "processing_log": [f"generation error: {exc}"],
            }

    def _generate_with_model(self, prompt: str, max_new_tokens: int, fast_mode: bool = False) -> str:
        """
        Generate a response from the underlying Tamil-LLaMA model.

        Args:
            prompt: Instruction-formatted prompt string.
            max_new_tokens: Maximum new tokens to decode.
            fast_mode: Whether to use lower-latency decoding settings.

        Returns:
            Decoded model response without the prompt prefix.
        """
        try:
            print("[TanglishBridgePipeline] Generating response with Tamil-LLaMA...")
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            generation_kwargs: Dict[str, object] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
            }
            if fast_mode:
                generation_kwargs["do_sample"] = False
            else:
                generation_kwargs["temperature"] = 0.7
                generation_kwargs["top_p"] = 0.9
                generation_kwargs["do_sample"] = True
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][prompt_length:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            if not decoded:
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            return decoded
        except Exception as exc:
            logger.exception("Model generation failed: %s", exc)
            return self._fallback_response(input_text=prompt, script_type="tanglish")

    def _fallback_response(self, input_text: str, script_type: str) -> str:
        """
        Produce a graceful fallback response when model inference is unavailable.

        Args:
            input_text: Original or prompt-formatted input.
            script_type: Detected input style.

        Returns:
            A short heuristic response matching the user's style.
        """
        try:
            print("[TanglishBridgePipeline] Building fallback response...")
            lowered = input_text.lower()
            if "name" in lowered or "பெயர்" in input_text:
                fallback = {
                    "english": "My name is TanglishBridge, and I am here to help.",
                    "tamil": "என் பெயர் TanglishBridge. நான் உதவ தயாராக இருக்கிறேன்.",
                    "romanized": "en peyar TanglishBridge. naan unga help panna ready.",
                    "tanglish": "En peyar TanglishBridge. Naan help panna ready.",
                }
                return fallback.get(script_type, fallback["tanglish"])
            if "saptiya" in lowered or "saapta" in lowered:
                return "Sapdala na ippove poi sapdu. Nee saptiya?"
            if "eppadi" in lowered or "how are you" in lowered:
                return "Naan nalla irukken. Neenga eppadi irukkeenga?"
            if "தமிழ்" in input_text:
                return "தமிழ் ஒரு செழுமையான திராவிட மொழி. அது இலக்கியமும் கலாச்சாரமும் நிறைந்தது."
            if script_type == "romanized":
                return "unga input purinjiduchu. konjam innum detail-ah sonna nalla help panna mudiyum."
            if script_type == "english":
                return "I understood your message. Share a little more context and I can help further."
            if script_type == "tamil":
                return "உங்கள் கேள்வி புரிந்தது. இன்னும் கொஞ்சம் விவரம் சொன்னால் நன்றாக உதவ முடியும்."
            return "Input purinjiduchu. Konjam detail-ah sollunga, better-aa help pannuren."
        except Exception as exc:
            logger.exception("Fallback response generation failed: %s", exc)
            return "I am here to help."

    def batch_generate(
        self,
        inputs_list: List[str],
        max_new_tokens: int = 200,
        fast_mode: bool = False,
    ) -> List[Dict[str, object]]:
        """
        Process multiple inputs sequentially through the TanglishBridge pipeline.

        Args:
            inputs_list: List of user inputs.
            max_new_tokens: Maximum new tokens per example.
            fast_mode: Whether to use lower-latency generation settings.

        Returns:
            A list of pipeline output dictionaries.
        """
        try:
            print("[TanglishBridgePipeline] Running batch generation...")
            return [
                self.generate(input_text=item, max_new_tokens=max_new_tokens, fast_mode=fast_mode)
                for item in inputs_list
            ]
        except Exception as exc:
            logger.exception("Batch generation failed: %s", exc)
            return []

    def get_pipeline_stats(self) -> Dict[str, object]:
        """
        Return aggregate usage statistics collected during this session.

        Returns:
            A dictionary containing counts, averages, and runtime device metadata.
        """
        try:
            print("[TanglishBridgePipeline] Returning pipeline statistics...")
            total_processed = int(self.stats["total_processed"])
            avg_cmi = round(self.stats["total_cmi"] / total_processed, 3) if total_processed else 0.0
            avg_response_length = round(self.stats["total_response_words"] / total_processed, 2) if total_processed else 0.0
            return {
                "total_processed": total_processed,
                "script_counts": dict(self.stats["script_counts"]),
                "average_cmi": avg_cmi,
                "average_response_length": avg_response_length,
                "model_name": self.stats["model_name"],
                "device": self.stats["device"],
                "model_available": self.model_available,
                "model_unavailable_reason": self.model_unavailable_reason,
            }
        except Exception as exc:
            logger.exception("Error while collecting pipeline stats: %s", exc)
            return {
                "total_processed": 0,
                "script_counts": {},
                "average_cmi": 0.0,
                "average_response_length": 0.0,
                "model_name": self.model_name,
                "device": self.device,
                "model_available": False,
                "model_unavailable_reason": self.model_unavailable_reason,
            }
