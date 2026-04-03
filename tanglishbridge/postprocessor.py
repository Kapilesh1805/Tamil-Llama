"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: postprocessor.py
Description: Response cleanup and style matching for Tamil-LLaMA generations.
"""

from __future__ import annotations

import logging
import re

from tanglishbridge.detector import ScriptDetector
from tanglishbridge.transliterator import RomanizedTamilTransliterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponsePostProcessor:
    """
    Cleans model output and aligns its presentation with the user's input style.
    """

    AI_BOILERPLATE_PATTERNS = [
        r"^i am an ai .*?$",
        r"^as an ai .*?$",
        r"^yes,\s*i am an ai .*?$",
    ]

    TANGISH_WORD_SWAPS = {
        "அலுவலகம்": "office",
        "சந்திப்பு": "meeting",
        "தொலைபேசி": "phone",
        "மடிக்கணினி": "laptop",
        "நன்றி": "thanks",
        "சரி": "okay",
        "வார இறுதி": "weekend",
    }

    BAD_ROMANIZATION_PATTERNS = [
        r"ghgh",
        r"bhbh",
        r"dhdh",
        r"jh",
        r"~",
        r"[a-z]{12,}",
    ]

    LIGHT_ROMANIZATION_MAP = {
        "வணக்கம்": "vanakkam",
        "நன்றி": "nandri",
        "சரி": "seri",
        "ஆம்": "aam",
        "இல்லை": "illai",
        "நான்": "naan",
        "நீ": "nee",
        "நீங்கள்": "neenga",
        "எப்படி": "eppadi",
        "எப்போது": "eppo",
        "சாப்பிட்டியா": "saptiya",
        "சாப்பிட்டாயா": "saptiya",
        "வருகிறேன்": "varen",
        "போகிறேன்": "poren",
        "இருக்கிறேன்": "irukken",
        "இருக்கிறது": "irukku",
        "வேண்டும்": "venum",
    }

    def __init__(self) -> None:
        """
        Initialize downstream response formatting helpers.
        """
        try:
            print("[ResponsePostProcessor] Initializing post-processor...")
            self.detector = ScriptDetector()
            self.transliterator = RomanizedTamilTransliterator()
        except Exception as exc:
            logger.exception("Failed to initialize ResponsePostProcessor: %s", exc)

    def process(self, response: str, input_style: str) -> str:
        """
        Clean and adapt a raw model response to the user's style.

        Args:
            response: Raw decoded model output.
            input_style: One of ``tanglish``, ``romanized``, ``tamil``, or ``english``.

        Returns:
            A cleaned response string ready for display.
        """
        try:
            print("[ResponsePostProcessor] Processing model response...")
            cleaned = self.clean_response(response)
            if input_style in {"tanglish", "romanized", "english"}:
                for pattern in self.AI_BOILERPLATE_PATTERNS:
                    cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

            cleaned = re.sub(r"\b(\w+)( \1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"([?.!,])\1+", r"\1", cleaned)

            sentences = [segment.strip() for segment in re.split(r"(?<=[.?!])\s+", cleaned) if segment.strip()]
            deduped_sentences = []
            for sentence in sentences:
                normalized_sentence = re.sub(r"\s+", " ", sentence.lower())
                if not deduped_sentences or re.sub(r"\s+", " ", deduped_sentences[-1].lower()) != normalized_sentence:
                    deduped_sentences.append(sentence)
            cleaned = " ".join(deduped_sentences) if deduped_sentences else cleaned

            words = cleaned.split()
            if len(words) > 150:
                cleaned = " ".join(words[:150]).strip()
                if cleaned and cleaned[-1] not in ".!?":
                    cleaned += "."

            if input_style == "romanized":
                romanized = self.transliterator.tamil_to_romanized(cleaned)
                if self._looks_like_bad_romanization(romanized):
                    cleaned = self._light_romanize_tamil_words(cleaned)
                else:
                    cleaned = romanized
            elif input_style == "tanglish" and self.detector.detect_script(cleaned) == "tamil":
                for tamil_word, english_word in self.TANGISH_WORD_SWAPS.items():
                    cleaned = cleaned.replace(tamil_word, english_word)
            elif input_style == "english" and self.detector.detect_script(cleaned) != "english":
                cleaned = "I understood your message. Let me answer that in English."

            if not cleaned.strip():
                fallback_map = {
                    "romanized": "naan inga irukken. unga kelviya konjam innum clear-ah sollunga.",
                    "tanglish": "I am here da. Konjam detail-ah sollunga, help pannuren.",
                    "english": "I am here to help. Please share a little more detail.",
                    "tamil": "நான் உதவ தயாராக இருக்கிறேன். உங்கள் கேள்வியை இன்னும் கொஞ்சம் தெளிவாக சொல்லுங்கள்.",
                }
                cleaned = fallback_map.get(input_style, "I am here to help.")

            return cleaned.strip()
        except Exception as exc:
            logger.exception("Error while post-processing response: %s", exc)
            return response.strip() or "I am here to help."

    def _looks_like_bad_romanization(self, text: str) -> bool:
        """
        Heuristically detect unusable Romanized output.

        Args:
            text: Romanized candidate string.

        Returns:
            ``True`` when the text looks garbled or unreadable.
        """
        try:
            print("[ResponsePostProcessor] Checking Romanization quality...")
            lowered = text.lower()
            if any(re.search(pattern, lowered) for pattern in self.BAD_ROMANIZATION_PATTERNS):
                return True
            return lowered.count("h") > max(6, len(lowered) // 5)
        except Exception as exc:
            logger.exception("Error while checking Romanization quality: %s", exc)
            return False

    def _light_romanize_tamil_words(self, text: str) -> str:
        """
        Apply a conservative Tamil-to-Romanized conversion for common response words.

        Args:
            text: Tamil-heavy response text.

        Returns:
            A lightly Romanized, more readable fallback string.
        """
        try:
            print("[ResponsePostProcessor] Applying light Romanization fallback...")
            light = text
            for tamil_word, romanized_word in self.LIGHT_ROMANIZATION_MAP.items():
                light = light.replace(tamil_word, romanized_word)
            return light
        except Exception as exc:
            logger.exception("Error while applying light Romanization fallback: %s", exc)
            return text

    def clean_response(self, text: str) -> str:
        """
        Remove prompt artifacts, redundant whitespace, and incomplete endings.

        Args:
            text: Raw decoded text.

        Returns:
            A cleaned response string.
        """
        try:
            print("[ResponsePostProcessor] Cleaning response text...")
            cleaned = text.replace("### Response:", "").replace("### Instruction:", "")
            cleaned = re.sub(r"^(assistant|response)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^["“”]+|["“”]+$', "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            if cleaned and cleaned[-1] not in ".!?":
                last_stop = max(cleaned.rfind("."), cleaned.rfind("?"), cleaned.rfind("!"))
                if last_stop > 20:
                    cleaned = cleaned[: last_stop + 1].strip()

            return cleaned
        except Exception as exc:
            logger.exception("Error while cleaning response text: %s", exc)
            return text.strip()

    def format_for_display(self, response: str) -> str:
        """
        Format a processed response for Streamlit rendering.

        Args:
            response: Response text after post-processing.

        Returns:
            A display-friendly string.
        """
        try:
            print("[ResponsePostProcessor] Formatting response for display...")
            formatted = response.strip()
            formatted = re.sub(r"\s+\n", "\n", formatted)
            formatted = re.sub(r"\n{3,}", "\n\n", formatted)
            return formatted
        except Exception as exc:
            logger.exception("Error while formatting response: %s", exc)
            return response
