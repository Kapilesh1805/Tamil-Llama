"""
TanglishBridge: Enabling Code-Mixed Tamil-English Interaction
for Tamil-LLaMA without Fine-Tuning
File: app.py
Description: Streamlit demo application for TanglishBridge inference and analysis.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tanglishbridge.pipeline import TanglishBridgePipeline


def load_json(path: str) -> Any:
    """
    Load a JSON file if it exists.

    Args:
        path: Relative or absolute JSON file path.

    Returns:
        Parsed JSON content or ``None``.
    """
    try:
        print("[app.py] Loading JSON resource...")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


@st.cache_resource(show_spinner=True)
def load_pipeline() -> TanglishBridgePipeline:
    """
    Cache the TanglishBridge pipeline for repeated UI interactions.

    Returns:
        Initialized pipeline instance.
    """
    try:
        print("[app.py] Loading TanglishBridge pipeline in Streamlit cache...")
        return TanglishBridgePipeline()
    except Exception as exc:
        st.error(f"Failed to initialize TanglishBridgePipeline: {exc}")
        raise


def render_badge(script_type: str) -> str:
    """
    Build a color-coded badge for the detected script type.

    Args:
        script_type: Detected script label.

    Returns:
        HTML string for badge rendering.
    """
    try:
        print("[app.py] Rendering script badge...")
        config = {
            "tamil": ("🔵 Tamil", "#1d4ed8"),
            "tanglish": ("🟢 Tanglish", "#0f766e"),
            "romanized": ("🟡 Romanized", "#ca8a04"),
            "english": ("⚪ English", "#475569"),
            "mixed": ("🟣 Mixed", "#7c3aed"),
        }
        label, color = config.get(script_type, ("⚪ Unknown", "#64748b"))
        return f"<span style='background:{color};color:white;padding:0.35rem 0.7rem;border-radius:999px;font-weight:600;'>{label}</span>"
    except Exception:
        return script_type


def main() -> None:
    """
    Streamlit entrypoint for the TanglishBridge demo.
    """
    try:
        print("[app.py] Starting Streamlit app...")
        st.set_page_config(page_title="TanglishBridge 🌉", page_icon="🌉", layout="wide")

        st.markdown(
            """
            <style>
                .stApp {
                    background: radial-gradient(circle at top left, #f7efe1 0%, #eef5db 48%, #d8e2dc 100%);
                }
                .hero-card, .output-card, .step-card {
                    background: rgba(255,255,255,0.82);
                    border: 1px solid rgba(15,118,110,0.15);
                    border-radius: 18px;
                    padding: 1rem 1.1rem;
                    box-shadow: 0 10px 30px rgba(31,41,55,0.08);
                }
                .pipeline-row {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.6rem;
                    align-items: center;
                    margin-top: 1rem;
                    margin-bottom: 1rem;
                }
                .step-chip {
                    background: linear-gradient(135deg, #0f766e, #1d4ed8);
                    color: white;
                    padding: 0.55rem 0.85rem;
                    border-radius: 999px;
                    font-size: 0.92rem;
                    font-weight: 600;
                }
                .arrow-chip {
                    color: #0f766e;
                    font-weight: 700;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        metrics_path = os.path.join(PROJECT_ROOT, "scripts", "eval", "metrics.json")
        cmi_stats_path = os.path.join(PROJECT_ROOT, "data", "processed", "cmi_stats.json")
        test_inputs_path = os.path.join(PROJECT_ROOT, "data", "processed", "test_inputs.json")
        histogram_path = os.path.join(PROJECT_ROOT, "data", "processed", "cmi_histogram.png")
        piechart_path = os.path.join(PROJECT_ROOT, "data", "processed", "cmi_piechart.png")

        metrics = load_json(metrics_path) or {}
        cmi_stats = load_json(cmi_stats_path) or {}
        test_inputs = load_json(test_inputs_path) or []

        with st.sidebar:
            st.markdown("## TanglishBridge v1.0")
            st.write("Model: Tamil-LLaMA 7B Instruct")
            st.write("Pipeline modules: 4")
            st.write("1. ScriptDetector")
            st.write("2. TanglishNormalizer")
            st.write("3. RomanizedTamilTransliterator")
            st.write("4. ResponsePostProcessor")
            st.write('Paper: "TanglishBridge: Enabling Code-Mixed..."')
            st.markdown("### Example prompts")
            example_prompts = [
                "bro saptiya?",
                "vanakkam, eppadi irukka?",
                "தமிழ் என்றால் என்ன?",
                "naan office ku late ah varen da",
                "tired bro, work romba iruku",
            ]
            for prompt in example_prompts:
                if st.button(prompt, width="stretch"):
                    st.session_state["tb_input"] = prompt

        st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
        st.title("TanglishBridge 🌉")
        st.caption("Bridging Tamil-English Code-Mixed Language for Tamil-LLaMA")
        st.markdown("</div>", unsafe_allow_html=True)

        pipeline = load_pipeline()
        tab_chat, tab_analysis = st.tabs(["💬 Chat", "📊 Analysis"])

        with tab_chat:
            if "tb_input" not in st.session_state:
                st.session_state["tb_input"] = "bro saptiya?"
            user_input = st.text_area(
                "Type in Tamil, English, Tanglish or Romanized Tamil...",
                height=120,
                key="tb_input",
            )

            if not pipeline.model_available:
                st.warning(
                    pipeline.model_unavailable_reason
                    or "Tamil-LLaMA could not be loaded in this environment, so the app is currently using fallback heuristic responses."
                )
            elif pipeline.device == "cpu":
                st.info(
                    "Tamil-LLaMA is running on CPU. Responses can be slow, so the app uses a smaller generation length by default."
                )

            fast_mode = st.toggle(
                "Fast mode",
                value=True if pipeline.device == "cpu" else False,
                help="Uses shorter, lower-latency generation settings for quicker responses.",
            )

            if fast_mode:
                st.caption("Fast mode keeps responses shorter and uses lower-latency decoding.")

            default_tokens = 24 if fast_mode and pipeline.device == "cpu" else 48 if pipeline.device == "cpu" else 64 if fast_mode else 160
            max_new_tokens = st.slider(
                "Response length",
                min_value=16,
                max_value=256,
                value=default_tokens,
                step=16,
            )

            if st.button("Submit", type="primary", width="stretch"):
                with st.spinner("TanglishBridge is processing your input..."):
                    result = pipeline.generate(
                        user_input,
                        max_new_tokens=max_new_tokens,
                        fast_mode=fast_mode,
                    )
                st.session_state["last_result"] = result

            if "last_result" in st.session_state:
                result = st.session_state["last_result"]
                st.markdown("<div class='pipeline-row'>", unsafe_allow_html=True)
                steps = ["Input", "Script Detection", "Normalization", "Transliteration", "Tamil-LLaMA", "Post-processing", "Output"]
                for index, step in enumerate(steps):
                    st.markdown(f"<span class='step-chip'>{step}</span>", unsafe_allow_html=True)
                    if index < len(steps) - 1:
                        st.markdown("<span class='arrow-chip'>→</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(render_badge(result["detected_script"]), unsafe_allow_html=True)
                    st.write("")
                    st.progress(min(float(result["cmi_score"]), 1.0))
                    st.caption(f"CMI Score: {result['cmi_score']:.3f}")
                with col2:
                    st.markdown("<div class='output-card'>", unsafe_allow_html=True)
                    st.subheader("Final Response")
                    st.write(result["final_response"])
                    st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("Processing Log", expanded=False):
                    for item in result["processing_log"]:
                        st.write(f"- {item}")

                st.markdown("### Intermediate View")
                intermediate_df = pd.DataFrame(
                    [
                        {"Stage": "Input", "Value": result["input"]},
                        {"Stage": "Normalized", "Value": result["normalized_input"]},
                        {"Stage": "Model Input", "Value": result["model_input"]},
                        {"Stage": "Raw Response", "Value": result["raw_response"]},
                    ]
                )
                st.dataframe(intermediate_df, width="stretch", hide_index=True)

        with tab_analysis:
            st.subheader("CMI Distribution")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                if os.path.exists(histogram_path):
                    st.image(histogram_path, width="stretch")
                else:
                    st.info("Run `python scripts/eval/cmi_analysis.py` to generate the histogram.")
            with chart_col2:
                if os.path.exists(piechart_path):
                    st.image(piechart_path, width="stretch")
                else:
                    st.info("Run `python scripts/eval/cmi_analysis.py` to generate the pie chart.")

            st.subheader("Example Inputs by Category")
            if test_inputs:
                example_df = pd.DataFrame(test_inputs)[["id", "script_type", "input", "reference_response"]]
                st.dataframe(example_df, width="stretch", hide_index=True)
            else:
                st.warning("No curated examples found in `data/processed/test_inputs.json`.")

            st.subheader("Before / After Comparison")
            if metrics.get("results"):
                comparison_rows: List[Dict[str, Any]] = []
                for item in metrics["results"]:
                    comparison_rows.append(
                        {
                            "Input": item["input"],
                            "Baseline": item["baseline_response"],
                            "TanglishBridge": item["ours_response"],
                        }
                    )
                st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
            else:
                st.info("Run `python scripts/eval/evaluate.py` to populate `metrics.json` for comparison.")

            st.subheader("Saved Metrics")
            if metrics:
                summary = {
                    "Script Accuracy (Baseline)": metrics.get("script_acc_baseline"),
                    "Script Accuracy (Ours)": metrics.get("script_acc_ours"),
                    "Coherence (Baseline)": metrics.get("coherence_baseline"),
                    "Coherence (Ours)": metrics.get("coherence_ours"),
                    "CMI Preservation (Baseline)": metrics.get("cmi_preservation_baseline"),
                    "CMI Preservation (Ours)": metrics.get("cmi_preservation_ours"),
                }
                st.json(summary)
            else:
                st.info("No metrics available yet. Evaluation results will appear here after running the evaluator.")

            if cmi_stats:
                st.subheader("CMI Snapshot")
                st.json(cmi_stats)
    except Exception as exc:
        st.error(f"TanglishBridge app failed: {exc}")


if __name__ == "__main__":
    main()
