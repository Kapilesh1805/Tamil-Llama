"""
Kaggle-ready Tamil chatbot that uses a Tamil Hugging Face model for
Tamil generation and Gemini for Tanglish conversion.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: google-generativeai. Install it with `pip install google-generativeai`."
    ) from exc


MODEL_NAME = os.environ.get("TAMIL_MODEL_NAME", "abhinand/tamil-llama-7b-instruct-v0.1")
MAX_HISTORY_MESSAGES = 5
MAX_NEW_TOKENS = 128
MAX_TANGLISH_RETRIES = 3

chat_history: List[Dict[str, str]] = []


def configure_gemini() -> genai.GenerativeModel:
    """
    Load Gemini API key from Kaggle Secrets / environment and configure Gemini.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        print("Error: GEMINI_API_KEY not found in Kaggle Secrets / environment.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def load_tamil_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """
    Load the Tamil Hugging Face model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)

    model.eval()
    return tokenizer, model, device


def _build_tamil_prompt(text: str) -> str:
    """
    Build the Tamil prompt with the last 5 chat messages as memory.
    """
    recent_history = chat_history[-MAX_HISTORY_MESSAGES:]
    history_lines: List[str] = []

    for message in recent_history:
        speaker = "பயனர்" if message["role"] == "user" else "உதவியாளர்"
        history_lines.append(f"{speaker}: {message['content']}")

    history_block = "\n".join(history_lines)
    if history_block:
        history_block += "\n"

    return (
        "### Instruction:\n"
        "நீங்கள் நண்பனாக, இயல்பாக, குறுகிய தமிழில் பதில் சொல்லும் உதவியாளர். "
        "பயனர் கேள்விக்கு நேராக பதில் சொல்லுங்கள். விளக்கமாக பேச வேண்டாம்.\n\n"
        f"{history_block}"
        f"பயனர்: {text}\n"
        "உதவியாளர்:"
    )


def _clean_tamil_generation(text: str) -> str:
    """
    Clean raw Tamil model output without aggressive rejection logic.
    """
    cleaned = text.strip()

    for marker in (
        "### Response:",
        "### Instruction:",
        "User:",
        "Assistant:",
        "பயனர்:",
        "உதவியாளர்:",
    ):
        cleaned = cleaned.replace(marker, "")

    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return "சரி, சொல்லு."

    return cleaned


def generate_tamil_response(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
) -> str:
    """
    Generate a Tamil response using the Hugging Face model.
    """
    prompt = _build_tamil_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")

    if device == "cuda":
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if not decoded.strip():
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

    return _clean_tamil_generation(decoded)


def _contains_non_ascii(text: str) -> bool:
    """
    Check whether text contains any non-ASCII characters.
    """
    return any(ord(char) > 127 for char in text)


def _normalize_ascii_text(text: str) -> str:
    """
    Normalize output into plain ASCII-safe Tanglish.
    """
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\n": " ",
        "\r": " ",
        "\t": " ",
    }
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _build_tanglish_prompt(tamil_text: str) -> str:
    """
    Prompt Gemini to return only short natural Tanglish.
    """
    return f"""Convert the following Tamil sentence into natural Tanglish (Roman Tamil).

STRICT RULES:

* Output ONLY Tanglish
* DO NOT use Tamil script
* DO NOT explain anything
* Use casual conversational style (like WhatsApp chat)
* Keep it short and natural

Tamil: {tamil_text}

Tanglish:
"""


def convert_to_tanglish(tamil_text: str, gemini_model: genai.GenerativeModel) -> str:
    """
    Convert Tamil output into Tanglish using Gemini.
    This function never falls back to Tamil. It retries until it gets ASCII Tanglish,
    then uses an ASCII-safe fallback message if API conversion still fails.
    """
    last_output = ""

    for _ in range(MAX_TANGLISH_RETRIES):
        try:
            response = gemini_model.generate_content(_build_tanglish_prompt(tamil_text))
            output = (response.text or "").strip()
            output = _normalize_ascii_text(output)

            if output and not _contains_non_ascii(output):
                return output

            last_output = output
        except Exception as exc:  # pragma: no cover
            print(f"Gemini API error: {exc}")

    if last_output and not _contains_non_ascii(last_output):
        return last_output

    return "seri da, konjam issue iruku. innum oru thadava try pannu."


def _update_history(user_text: str, tamil_response: str) -> None:
    """
    Store only the last 5 chat messages for memory context.
    """
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": tamil_response})
    del chat_history[:-MAX_HISTORY_MESSAGES]


def chatbot() -> None:
    """
    Main chatbot loop.
    """
    gemini_model = configure_gemini()
    tokenizer, model, device = load_tamil_model()

    print("Tamil Tanglish chatbot ready. Type 'exit' to stop.")

    while True:
        user_text = input("User: ").strip()

        if not user_text:
            print("Bot (Tanglish): please message anuppu da.")
            continue

        if user_text.lower() in {"exit", "quit", "bye"}:
            print("Bot (Tanglish): seri bro, later pesalam.")
            break

        tamil_response = generate_tamil_response(user_text, tokenizer, model, device)
        tanglish_response = convert_to_tanglish(tamil_response, gemini_model)

        _update_history(user_text, tamil_response)

        print(f"Bot (Tanglish): {tanglish_response}")


if __name__ == "__main__":
    chatbot()
