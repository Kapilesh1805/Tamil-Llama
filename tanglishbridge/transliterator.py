"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: transliterator.py
Description: Smart transliteration helpers between Romanized Tamil and Tamil script.
"""

from __future__ import annotations

import logging
import re

from indic_transliteration import sanscript
from indic_transliteration.detect import detect
from indic_transliteration.sanscript import transliterate

from tanglishbridge.detector import ScriptDetector
from tanglishbridge.normalizer import TanglishNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RomanizedTamilTransliterator:
    """
    Converts Romanized Tamil to Tamil script while preserving English context words.
    """

    SPECIAL_ROMANIZED_MAPPINGS = {
        "vanakkam": "வணக்கம்",
        "nandri": "நன்றி",
        "naan": "நான்",
        "nan": "நான்",
        "nee": "நீ",
        "ni": "நீ",
        "neenga": "நீங்கள்",
        "enna": "என்ன",
        "ennada": "என்னடா",
        "epdi": "எப்படி",
        "eppadi": "எப்படி",
        "eppo": "எப்போது",
        "ippo": "இப்போது",
        "inga": "இங்கு",
        "anga": "அங்கு",
        "romba": "ரொம்ப",
        "konjam": "கொஞ்சம்",
        "nalla": "நல்ல",
        "seri": "சரி",
        "sari": "சரி",
        "illa": "இல்லை",
        "illai": "இல்லை",
        "varen": "வருகிறேன்",
        "varren": "வருகிறேன்",
        "poren": "போகிறேன்",
        "porren": "போகிறேன்",
        "saapta": "சாப்பிட்டா",
        "sapta": "சாப்பிட்டா",
        "saptiya": "சாப்பிட்டியா",
        "saaptiya": "சாப்பிட்டியா",
        "theriyala": "தெரியவில்லை",
        "puriyala": "புரியவில்லை",
        "venum": "வேண்டும்",
        "venam": "வேண்டாம்",
        "aaguthu": "ஆகுது",
        "aagum": "ஆகும்",
        "padam": "படம்",
        "paakalam": "பாக்கலாம்",
        "pogalam": "போகலாம்",
        "vaa": "வா",
        "vaanga": "வாங்க",
        "sollu": "சொல்லு",
        "paaru": "பாரு",
        "mokka": "மொக்கை",
        "semma": "செம்ம",
    }

    REVERSE_ROMANIZED_MAPPINGS = {value: key for key, value in SPECIAL_ROMANIZED_MAPPINGS.items()}

    KEEP_AS_IS_WORDS = {
        "bro",
        "da",
        "di",
        "ku",
        "ah",
        "la",
        "super",
        "office",
        "meeting",
        "phone",
        "laptop",
        "coffee",
        "tea",
        "traffic",
        "exam",
        "result",
        "weekend",
        "plan",
        "work",
        "project",
        "deadline",
        "report",
        "call",
        "message",
        "number",
        "mail",
        "zoom",
        "rest",
        "sleep",
        "rain",
        "umbrella",
        "congrats",
        "beach",
        "movie",
    }

    def __init__(self) -> None:
        """
        Initialize transliteration helpers and lexical resources.
        """
        try:
            print("[RomanizedTamilTransliterator] Initializing transliterator...")
            self.detector = ScriptDetector()
            self.normalizer = TanglishNormalizer()
        except Exception as exc:
            logger.exception("Failed to initialize transliterator: %s", exc)

    def romanized_to_tamil(self, text: str) -> str:
        """
        Convert Romanized Tamil words to Tamil script while preserving English words.

        Args:
            text: Romanized or mixed input text.

        Returns:
            Tamil-script-enriched text suitable for Tamil-LLaMA prompting.
        """
        try:
            print("[RomanizedTamilTransliterator] Converting Romanized Tamil to Tamil script...")
            tokens = re.findall(r"[\u0B80-\u0BFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text, flags=re.UNICODE)
            converted_tokens = []

            for token in tokens:
                if re.fullmatch(r"[^\w\s]", token):
                    converted_tokens.append(token)
                    continue

                lowered = token.lower()
                if any("\u0B80" <= ch <= "\u0BFF" for ch in token):
                    converted_tokens.append(token)
                elif lowered in self.KEEP_AS_IS_WORDS:
                    converted_tokens.append(lowered)
                elif lowered in self.SPECIAL_ROMANIZED_MAPPINGS:
                    converted_tokens.append(self.SPECIAL_ROMANIZED_MAPPINGS[lowered])
                elif lowered in self.normalizer.VERB_MAPPINGS:
                    converted_tokens.append(self.normalizer.VERB_MAPPINGS[lowered])
                elif self.is_romanized_tamil(lowered):
                    try:
                        detected_script = detect(lowered)
                        logger.info("Indic transliteration detect(%s) -> %s", lowered, detected_script)
                    except Exception:
                        logger.info("Indic detection unavailable for token: %s", lowered)
                    try:
                        converted_tokens.append(transliterate(lowered, sanscript.ITRANS, sanscript.TAMIL))
                    except Exception:
                        logger.info("Fallback transliteration failed for %s; keeping token as-is.", lowered)
                        converted_tokens.append(token)
                else:
                    converted_tokens.append(lowered)

            converted_text = " ".join(converted_tokens)
            converted_text = re.sub(r"\s+([?.!,;:])", r"\1", converted_text)
            converted_text = re.sub(r"\s{2,}", " ", converted_text).strip()
            return converted_text
        except Exception as exc:
            logger.exception("Error while converting Romanized Tamil to Tamil: %s", exc)
            return text

    def tamil_to_romanized(self, text: str) -> str:
        """
        Convert Tamil script back into Romanized Tamil for style matching.

        Args:
            text: Tamil-script response text.

        Returns:
            A readable Romanized form of the Tamil text.
        """
        try:
            print("[RomanizedTamilTransliterator] Converting Tamil script to Romanized Tamil...")
            tokens = re.findall(r"[\u0B80-\u0BFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text, flags=re.UNICODE)
            romanized_tokens = []

            for token in tokens:
                if re.fullmatch(r"[^\w\s]", token):
                    romanized_tokens.append(token)
                    continue

                if any("\u0B80" <= ch <= "\u0BFF" for ch in token):
                    if token in self.REVERSE_ROMANIZED_MAPPINGS:
                        romanized = self.REVERSE_ROMANIZED_MAPPINGS[token]
                    else:
                        romanized = transliterate(token, sanscript.TAMIL, sanscript.ITRANS)
                        romanized = romanized.replace("N", "n").replace("T", "t").replace("D", "d").replace("L", "l")
                        romanized = romanized.replace("RR", "r").replace("Sh", "sh").lower()
                    romanized_tokens.append(romanized)
                else:
                    romanized_tokens.append(token.lower())

            romanized_text = " ".join(romanized_tokens)
            romanized_text = re.sub(r"\s+([?.!,;:])", r"\1", romanized_text)
            romanized_text = re.sub(r"\s{2,}", " ", romanized_text).strip()
            return romanized_text
        except Exception as exc:
            logger.exception("Error while converting Tamil to Romanized form: %s", exc)
            return text

    def is_romanized_tamil(self, word: str) -> bool:
        """
        Check whether a token resembles Romanized Tamil instead of English.

        Args:
            word: Input token to classify.

        Returns:
            ``True`` when the token matches Tamil lexical or phonetic patterns.
        """
        try:
            print("[RomanizedTamilTransliterator] Checking whether token is Romanized Tamil...")
            cleaned = re.sub(r"[^A-Za-z]", "", word).lower()
            if not cleaned or any("\u0B80" <= ch <= "\u0BFF" for ch in cleaned):
                return False
            if cleaned in self.KEEP_AS_IS_WORDS:
                return False
            if cleaned in self.SPECIAL_ROMANIZED_MAPPINGS or cleaned in self.normalizer.VERB_MAPPINGS:
                return True
            if cleaned in self.detector.ROMANIZED_TAMIL_WORDS and cleaned not in self.detector.CONTEXTUAL_ENGLISH_WORDS:
                return True
            tamil_like_pattern = re.search(r"(aa|ee|oo|zh|ng|th|dh|kk|pp|tt|rr|ll|nn|ai|au)", cleaned)
            tamil_like_ending = re.search(r"(ren|rom|nga|num|la|ku|da|di|ya|tha|lam|ttu|cha|um|ala|iya)$", cleaned)
            return bool(tamil_like_pattern and tamil_like_ending)
        except Exception as exc:
            logger.exception("Error while checking Romanized Tamil token: %s", exc)
            return False

    def smart_transliterate(self, text: str) -> str:
        """
        Transliterate only the tokens that should become Tamil script.

        Args:
            text: Mixed Tanglish or Romanized Tamil text.

        Returns:
            A best-effort transliterated string for the model.
        """
        try:
            print("[RomanizedTamilTransliterator] Performing smart transliteration...")
            detected_style = self.detector.detect_script(text)
            if detected_style == "english":
                return text
            return self.romanized_to_tamil(text)
        except Exception as exc:
            logger.exception("Error during smart transliteration: %s", exc)
            return text
