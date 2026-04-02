"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: normalizer.py
Description: Normalization utilities for Tanglish and Romanized Tamil input text.
"""

from __future__ import annotations

import logging
import re
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TanglishNormalizer:
    """
    Expands abbreviations and maps frequent Romanized Tamil forms into model-friendly text.
    """

    ABBREVIATIONS = {
        "r": "are",
        "u": "you",
        "ur": "your",
        "urs": "yours",
        "da": "(informal address)",
        "di": "(informal address female)",
        "bro": "brother",
        "macha": "friend",
        "machi": "friend",
        "anna": "elder brother",
        "akka": "elder sister",
        "thambi": "younger brother",
        "pls": "please",
        "plsda": "please (informal address)",
        "plz": "please",
        "tmrw": "tomorrow",
        "tmr": "tomorrow",
        "tdy": "today",
        "msg": "message",
        "num": "number",
        "wt": "what",
        "hw": "how",
        "whr": "where",
        "wer": "where",
        "y": "why",
        "b4": "before",
        "abt": "about",
        "idk": "i do not know",
        "imo": "in my opinion",
        "imho": "in my honest opinion",
        "asap": "as soon as possible",
        "fyi": "for your information",
        "thx": "thanks",
        "ty": "thank you",
        "sry": "sorry",
        "btw": "by the way",
        "bcz": "because",
        "coz": "because",
        "cuz": "because",
        "frnd": "friend",
        "frnds": "friends",
        "luv": "love",
        "gn": "good night",
        "gm": "good morning",
        "ge": "good evening",
        "nyt": "night",
        "evng": "evening",
        "mrng": "morning",
        "wknd": "weekend",
        "prob": "problem",
        "dept": "department",
        "mins": "minutes",
        "min": "minute",
        "hrs": "hours",
        "hr": "hour",
        "sec": "second",
        "approx": "approximately",
        "info": "information",
        "bsy": "busy",
        "ok": "okay",
        "k": "okay",
        "gr8": "great",
        "gud": "good",
        "vl": "will",
        "wl": "will",
        "omw": "on my way",
        "brb": "be right back",
        "ttyl": "talk to you later",
        "afaik": "as far as i know",
        "msgme": "message me",
        "callme": "call me",
    }

    VERB_MAPPINGS = {
        "pannren": "செய்கிறேன்",
        "pannrom": "செய்கிறோம்",
        "pannita": "செய்துவிட்டேன்",
        "pannitan": "செய்துவிட்டான்",
        "pannunga": "செய்யுங்கள்",
        "irukken": "இருக்கிறேன்",
        "irukkom": "இருக்கிறோம்",
        "irukku": "இருக்கிறது",
        "irukka": "இருக்கிறாயா",
        "solren": "சொல்கிறேன்",
        "sollren": "சொல்கிறேன்",
        "sollu": "சொல்",
        "varren": "வருகிறேன்",
        "varen": "வருகிறேன்",
        "varom": "வருகிறோம்",
        "porren": "போகிறேன்",
        "poren": "போகிறேன்",
        "poganum": "போக வேண்டும்",
        "varanum": "வர வேண்டும்",
        "sapten": "சாப்பிட்டேன்",
        "saapten": "சாப்பிட்டேன்",
        "sapta": "சாப்பிட்டாயா",
        "saapta": "சாப்பிட்டாயா",
        "sapdran": "சாப்பிடுகிறான்",
        "saapdran": "சாப்பிடுகிறான்",
        "kekkuren": "கேட்கிறேன்",
        "kekuren": "கேட்கிறேன்",
        "paakren": "பார்க்கிறேன்",
        "paakrom": "பார்க்கிறோம்",
        "pesren": "பேசுகிறேன்",
        "pesrom": "பேசுகிறோம்",
        "theriyala": "தெரியவில்லை",
        "therila": "தெரியவில்லை",
        "puriyala": "புரியவில்லை",
        "puriala": "புரியவில்லை",
        "venum": "வேண்டும்",
        "venuma": "வேண்டுமா",
        "venam": "வேண்டாம்",
        "mudiyala": "முடியவில்லை",
        "mudila": "முடியவில்லை",
        "aaguthu": "ஆகிறது",
        "aagum": "ஆகும்",
        "vandhen": "வந்தேன்",
        "vandhuten": "வந்துவிட்டேன்",
        "padikren": "படிக்கிறேன்",
        "padikrom": "படிக்கிறோம்",
        "edukanum": "எடுக்க வேண்டும்",
        "kudukanum": "கொடுக்க வேண்டும்",
    }

    TECHNICAL_WORDS = {
        "office",
        "meeting",
        "phone",
        "laptop",
        "coffee",
        "tea",
        "project",
        "deadline",
        "result",
        "exam",
        "traffic",
        "weekend",
        "plan",
        "work",
        "message",
        "number",
        "call",
        "mail",
        "zoom",
        "report",
        "movie",
        "beach",
        "rest",
        "sleep",
        "congrats",
        "rain",
        "umbrella",
    }

    def __init__(self) -> None:
        """
        Initialize stateful normalization history.
        """
        try:
            print("[TanglishNormalizer] Initializing normalizer...")
            self.last_log: List[str] = []
        except Exception as exc:
            logger.exception("Failed to initialize TanglishNormalizer: %s", exc)
            self.last_log = []

    def normalize(self, text: str) -> str:
        """
        Normalize Tanglish input to a Tamil-LLaMA-friendly form.

        Args:
            text: Raw user input.

        Returns:
            A normalized string with expanded abbreviations and mapped verbs.
        """
        try:
            print("[TanglishNormalizer] Normalizing input text...")
            self.last_log = []
            tokens = re.findall(r"[\u0B80-\u0BFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text, flags=re.UNICODE)
            normalized_tokens: List[str] = []

            for token in tokens:
                if re.fullmatch(r"[^\w\s]", token):
                    normalized_tokens.append(token)
                    continue

                lowered = token.lower()
                replacement = token

                if lowered in self.ABBREVIATIONS:
                    replacement = self.ABBREVIATIONS[lowered]
                    self.last_log.append(f"abbreviation: {token} -> {replacement}")
                elif lowered in self.VERB_MAPPINGS:
                    replacement = self.VERB_MAPPINGS[lowered]
                    self.last_log.append(f"verb mapping: {token} -> {replacement}")
                elif lowered in self.TECHNICAL_WORDS:
                    replacement = lowered
                elif lowered.isascii():
                    replacement = lowered

                normalized_tokens.append(replacement)

            normalized_text = " ".join(normalized_tokens)
            normalized_text = re.sub(r"\s+([?.!,;:])", r"\1", normalized_text)
            normalized_text = re.sub(r"\(\s+", "(", normalized_text)
            normalized_text = re.sub(r"\s+\)", ")", normalized_text)
            normalized_text = re.sub(r"\s{2,}", " ", normalized_text).strip()
            if not self.last_log:
                self.last_log.append("no normalization changes applied")
            logger.info("Normalization complete: %s", normalized_text)
            return normalized_text
        except Exception as exc:
            logger.exception("Error while normalizing text: %s", exc)
            return text

    def get_normalization_log(self, text: str) -> List[str]:
        """
        Return the explicit list of changes introduced by normalization.

        Args:
            text: Raw text to analyze.

        Returns:
            A list of human-readable transformation strings.
        """
        try:
            print("[TanglishNormalizer] Fetching normalization log...")
            self.normalize(text)
            return list(self.last_log)
        except Exception as exc:
            logger.exception("Error while building normalization log: %s", exc)
            return ["normalization log unavailable"]
