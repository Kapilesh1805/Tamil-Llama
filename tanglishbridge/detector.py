"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: detector.py
Description: Script and language detection utilities for Tamil, English, and Tanglish text.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScriptDetector:
    """
    Detects script type, token-level language, and code-mixing statistics.
    """

    ROMANIZED_TAMIL_WORDS = set(
        """
        vanakkam nandri sari seri seriinga illa illai illanga enna ennada ennanga enaku enakku unaku unakku
        enoda ennoda unoda ungaloda namma nammala namakku naan nan na nee ni neenga neengainga avan ava avanga
        ivan iva ivanga unga ungalukku appa amma anna akka thambi thangachi macha machi pa da di ma paati thatha
        mama mami chithi chithappa periyappa periyamma ammachi paiyan ponnu pasanga ponnunga veedu veetla veetukku
        inga ange anga inge enga enge engayo eppo ippo appo inniku inikku innaiku naalaiku nalaiku nethu naethu
        indha intha anda antha ithu adhu engeyo yenga eppadi epdi eppo yedhukku ethuku evlo evalo evvalo romba
        konjam nalla ketta semma summa vera apdi appdi appadi apdiye appdiye aparam aprom apram ipdi ippadi
        ippadiye sapdu saapdu saapta sapta saptiya saaptiya sapten saptena saapten saaptacha saaptachaam saapadu
        soru tiffen tea kaapi coffee paal thanni sapdran sapdranga saapdradhu kudichiya kudichiyaa kudika kudikanum
        varen varren varuvan varuva varuvaa varala varanum vandhen vandha vandhuten vandhutta vandhirukken vandhiruka
        poren porren poga poganum poitu poittu poita poiten poi po poonga poidu vaa vaanga vanga vaada vaadi vandhudu
        sollu solu solren sollren solra sonna ketka kelu kelu paathu paaru parunga paakalam paakren paakrom paarthen
        pakka paakalama pannu pannu pannren pannrom pannita pannitan pannalama pannuvom pannunga iruku irukku irukken
        irukkom iruka irukka irukkum irundha irundhuchu theriyum theriyala therila therinjidum puriyum puriyala puriala
        pidikkum pidikala pidikkala venum venuma venam venamda aagum aaguthu aachu mudiyuma mudiyala mudila kidaikkuma
        edukanum edukanum edutukitiya eduthu kudukanum kudukren kudutha vaangiko vaanginen padikren padikrom padikanum
        padikala pesu paesu pesren pesrom paesalama sirikkuthu sirippu bayama kastama kashtama santhosham sogama tiredah
        lateah easyah busyah correctah nallaa kettaa semmaya supera massa sceneu gethu mokka overa lighta stronga
        officeku meetingku veetukku schoolku collegela busla trainla roadla trafficla examla resultu marku passu failu
        padam paatu beachu planu worku velai velaila phoneu laptopu chargeru numberu messageu callu zoomu mailu
        projectu reportu targetu deadlineu leaveu lunchu dinneru breakfastu snacku restu sleepu nightu morningu eveningu
        umbrella rainu mazhai aaguthu varutha varudha theriyuma theriyutha kekuren kekkuren kekrom kelunga paakuriya
        poyacha vandhacha aayiducha nadakudhu nadakuthu irundhutu ukkaru nillu vaainga pogalaam varalaam polama pakalama
        sollalama kekkalama paesalama sandhoshama kovama sirichiya azhuthiya yosichiya purinjiducha purinjutha
        """
        .split()
    )

    CONTEXTUAL_ENGLISH_WORDS = {
        "bro",
        "super",
        "mass",
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
        "report",
        "deadline",
        "message",
        "number",
        "call",
        "zoom",
        "mail",
        "beach",
        "movie",
        "padam",
        "congrats",
        "rest",
        "sleep",
        "rain",
        "bus",
        "train",
    }

    def __init__(self) -> None:
        """
        Initialize reusable detector resources.
        """
        try:
            print("[ScriptDetector] Initializing detector resources...")
            logger.info("Loaded %s romanized Tamil reference words.", len(self.ROMANIZED_TAMIL_WORDS))
        except Exception as exc:
            logger.exception("Failed to initialize ScriptDetector: %s", exc)

    def detect_script(self, text: str) -> str:
        """
        Detect the dominant script style for a text snippet.

        Args:
            text: Input text from the user.

        Returns:
            One of ``tamil``, ``english``, ``tanglish``, ``romanized``, or ``mixed``.
        """
        try:
            print("[ScriptDetector] Detecting script type...")
            tokens: List[str] = re.findall(r"[\u0B80-\u0BFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
            has_tamil_script = any(any("\u0B80" <= ch <= "\u0BFF" for ch in token) for token in tokens)
            romanized_count = 0
            english_count = 0

            for token in tokens:
                cleaned = token.strip().lower()
                if not cleaned:
                    continue
                if any("\u0B80" <= ch <= "\u0BFF" for ch in cleaned):
                    continue
                if cleaned in self.CONTEXTUAL_ENGLISH_WORDS:
                    english_count += 1
                    continue
                if cleaned in self.ROMANIZED_TAMIL_WORDS:
                    romanized_count += 1
                    continue
                if cleaned.isascii():
                    english_count += 1

            if not tokens:
                return "mixed"
            if has_tamil_script and romanized_count == 0 and english_count == 0:
                return "tamil"
            if not has_tamil_script and romanized_count > 0 and english_count == 0:
                return "romanized"
            if romanized_count > 0 and english_count > 0 and not has_tamil_script:
                return "tanglish"
            if has_tamil_script and english_count > 0 and romanized_count == 0:
                return "tanglish"
            if has_tamil_script and (english_count > 0 or romanized_count > 0):
                return "mixed"
            if english_count > 0 and romanized_count == 0 and not has_tamil_script:
                return "english"
            return "mixed"
        except Exception as exc:
            logger.exception("Error while detecting script: %s", exc)
            return "mixed"

    def detect_word_language(self, word: str) -> str:
        """
        Detect a token's literal script language.

        Args:
            word: A single token to analyze.

        Returns:
            ``tamil`` if Tamil Unicode is present, ``english`` if pure ASCII, else ``unknown``.
        """
        try:
            print("[ScriptDetector] Detecting token-level language...")
            cleaned = word.strip()
            if not cleaned:
                return "unknown"
            if any("\u0B80" <= ch <= "\u0BFF" for ch in cleaned):
                return "tamil"
            if cleaned.isascii():
                return "english"
            return "unknown"
        except Exception as exc:
            logger.exception("Error while detecting word language: %s", exc)
            return "unknown"

    def calculate_cmi(self, text: str) -> float:
        """
        Compute Code-Mixing Index (CMI) using semantic Tamil and English token counts.

        Args:
            text: Input text to score.

        Returns:
            A float in the range ``[0, 1]``.
        """
        try:
            print("[ScriptDetector] Calculating CMI...")
            tokens: List[str] = re.findall(r"[\u0B80-\u0BFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
            total_words = len(tokens)
            if total_words == 0:
                return 0.0

            tamil_words = 0
            english_words = 0

            for token in tokens:
                cleaned = token.lower()
                if any("\u0B80" <= ch <= "\u0BFF" for ch in cleaned):
                    tamil_words += 1
                elif cleaned in self.CONTEXTUAL_ENGLISH_WORDS:
                    english_words += 1
                elif cleaned in self.ROMANIZED_TAMIL_WORDS:
                    tamil_words += 1
                elif cleaned.isascii():
                    english_words += 1

            max_li = max(tamil_words, english_words)
            if tamil_words == 0 or english_words == 0:
                return 0.0
            cmi_score = (total_words - max_li) / total_words
            return round(max(0.0, min(1.0, cmi_score)), 3)
        except Exception as exc:
            logger.exception("Error while calculating CMI: %s", exc)
            return 0.0

    def get_text_stats(self, text: str) -> Dict[str, object]:
        """
        Gather script and language statistics for a text snippet.

        Args:
            text: Input text to summarize.

        Returns:
            A dictionary with script type, counts, CMI score, and romanization flag.
        """
        try:
            print("[ScriptDetector] Building text statistics...")
            tokens: List[str] = re.findall(r"[\u0B80-\u0BFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
            tamil_words = 0
            english_words = 0
            has_romanized = False

            for token in tokens:
                cleaned = token.lower()
                if any("\u0B80" <= ch <= "\u0BFF" for ch in cleaned):
                    tamil_words += 1
                elif cleaned in self.CONTEXTUAL_ENGLISH_WORDS:
                    english_words += 1
                elif cleaned in self.ROMANIZED_TAMIL_WORDS:
                    tamil_words += 1
                    has_romanized = True
                elif cleaned.isascii():
                    english_words += 1

            return {
                "script_type": self.detect_script(text),
                "total_words": len(tokens),
                "tamil_words": tamil_words,
                "english_words": english_words,
                "cmi_score": self.calculate_cmi(text),
                "has_romanized": has_romanized,
            }
        except Exception as exc:
            logger.exception("Error while collecting text stats: %s", exc)
            return {
                "script_type": "mixed",
                "total_words": 0,
                "tamil_words": 0,
                "english_words": 0,
                "cmi_score": 0.0,
                "has_romanized": False,
            }
