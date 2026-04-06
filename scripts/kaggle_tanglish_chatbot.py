"""
Kaggle-ready Tamil chatbot that uses a Tamil Hugging Face model for
Tamil generation and Gemini for Tanglish conversion.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "Missing dependency: google-generativeai. Install it with "
        "`pip install google-generativeai` in your Kaggle notebook."
    ) from exc


MODEL_NAME = os.environ.get("TAMIL_MODEL_NAME", "abhinand/tamil-llama-7b-instruct-v0.1")
MAX_HISTORY_MESSAGES = 5
MAX_NEW_TOKENS = 128

# In-memory chat history, storing the last 5 messages.
chat_history: List[Dict[str, str]] = []


def configure_gemini() -> genai.GenerativeModel:
    """
    Load Gemini API key from Kaggle Secrets and configure the client.

    Returns:
        A configured Gemini GenerativeModel instance.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        print("Error: GEMINI_API_KEY not found in Kaggle Secrets / environment.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def load_tamil_model():
    """
    Load tokenizer and Tamil generation model from Hugging Face.

    Returns:
        Tuple of tokenizer, model, and resolved device string.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    return tokenizer, model, device


def _build_tamil_prompt(text: str) -> str:
    """
    Build the Tamil model prompt using recent chat memory.

    Args:
        text: Latest user message.

    Returns:
        Prompt string for the Tamil language model.
    """
    recent_history = chat_history[-MAX_HISTORY_MESSAGES:]
    history_lines = []
    for message in recent_history:
        prefix = "பயனர்" if message["role"] == "user" else "உதவியாளர்"
        history_lines.append(f"{prefix}: {message['content']}")

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


def _clean_generated_text(text: str) -> str:
    """
    Clean the raw Tamil model output.

    Args:
        text: Raw generated text.

    Returns:
        Cleaned Tamil response.
    """
    cleaned = text.strip()
    for marker in ("### Response:", "### Instruction:", "பயனர்:", "User:", "Assistant:"):
        cleaned = cleaned.replace(marker, "")
    cleaned = cleaned.strip().strip('"').strip("'")

    if cleaned and cleaned[-1] not in ".?!":
        last_stop = max(cleaned.rfind("."), cleaned.rfind("?"), cleaned.rfind("!"))
        if last_stop > 0:
            cleaned = cleaned[: last_stop + 1].strip()

    return cleaned or "சரி, சொல்லு."


def generate_tamil_response(text: str, tokenizer, model, device: str) -> str:
    """
    Generate a Tamil response using the Hugging Face model.

    Args:
        text: User input.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face causal LM.
        device: Active device name.

    Returns:
        Tamil model response.
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
    return _clean_generated_text(decoded)


def convert_to_tanglish(tamil_text: str, gemini_model: genai.GenerativeModel) -> str:
    """
    Convert Tamil output into casual Tanglish using Gemini.

    Args:
        tamil_text: Tamil response text.
        gemini_model: Configured Gemini model.

    Returns:
        Tanglish response text.
    """
    prompt = (
        "Convert the following Tamil sentence into natural Tanglish used in casual chat. "
        "Avoid formal transliteration.\n\n"
        f"Tamil: {tamil_text}\n"
        "Tanglish:"
    )

    try:
        response = gemini_model.generate_content(prompt)
        tanglish_text = (response.text or "").strip()
        return tanglish_text or tamil_text
    except Exception as exc:  # pragma: no cover - API/network specific
        print(f"Gemini API error: {exc}")
        return tamil_text


def _update_history(user_text: str, tamil_text: str) -> None:
    """
    Update the rolling chat history and keep only the last 5 messages.

    Args:
        user_text: Latest user message.
        tamil_text: Latest Tamil model response.
    """
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": tamil_text})
    del chat_history[:-MAX_HISTORY_MESSAGES]


def chatbot() -> None:
    """
    Run the interactive chatbot loop.
    """
    gemini_model = configure_gemini()
    tokenizer, model, device = load_tamil_model()

    print("Tamil Tanglish chatbot ready. Type 'exit' to stop.")

    while True:
        user_text = input("User: ").strip()

        if not user_text:
            print("Bot (Tanglish): Please enter a message.")
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
