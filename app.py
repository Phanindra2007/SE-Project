"""Streamlit UI for local banking requirements and SDLC generation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.step1_generate_reqs import generate_requirements
from src.step2_select_sdlc import select_sdlc


DATA_DIR = PROJECT_ROOT / "data" / "banking"
STEP1_OUTPUT = PROJECT_ROOT / "outputs" / "step1_requirements.json"
STEP2_JSON = PROJECT_ROOT / "outputs" / "step2_sdlc_justification.json"
STEP2_MARKDOWN = PROJECT_ROOT / "outputs" / "step2_sdlc_justification.md"


def init_state() -> None:
    st.session_state.setdefault("is_running", False)
    st.session_state.setdefault("step1_output", None)
    st.session_state.setdefault("step2_output", None)
    st.session_state.setdefault("pending_step", None)
    st.session_state.setdefault("pending_functionality", "")
    st.session_state.setdefault("pending_context", None)
    st.session_state.setdefault("error_message", None)


def explain_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, FileNotFoundError):
        return f"Missing file: {message}"
    if isinstance(exc, json.JSONDecodeError):
        return f"Invalid JSON: {message}"
    if "cannot reach" in lowered or "connection" in lowered or "connect" in lowered:
        return (
            f"Cannot reach the local Ollama/OpenAI-compatible server at "
            f"{os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1/')}. "
            "Start Ollama, verify the model name, and try again."
        )
    if isinstance(exc, ValueError):
        return f"Invalid JSON or schema: {message}"
    return f"Unexpected error: {message}"


def format_fit(value: int | float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def render_step1(output: dict[str, Any]) -> None:
    profile = output.get("functionality_profile")
    if isinstance(profile, dict):
        st.subheader("Functionality Profile")
        target = str(profile["target_functionality"])
        st.markdown(f"**Target Functionality:** {target}")

        columns = st.columns(2)
        columns[0].metric("Criticality", str(profile["estimated_criticality"]))
        columns[1].metric("Change Frequency", str(profile["expected_change_frequency"]))

    requirements = output.get("requirements", [])
    st.subheader(f"Generated Requirements ({len(requirements)})")

    for item in requirements:
        if not isinstance(item, dict):
            continue
        with st.container(border=True):
            st.markdown(
                f"**{item['id']}** "
                f"`{item['type']}`"
            )
            st.markdown(f"### {item['title']}")
            st.write(item["requirement"])
            reasoning = str(item["reasoning"])
            if "[Assumption]" in reasoning:
                st.warning(reasoning)
            else:
                st.caption(reasoning)
            references = item["source_references"]
            if isinstance(references, list) and references:
                st.markdown("**Source references:**")
                for reference in references:
                    st.markdown(f"- `{reference}`")

    with st.expander("Raw Step 1 JSON", expanded=False):
        st.json(output)


def render_step2(output: dict[str, Any]) -> None:
    st.metric("Recommended SDLC", str(output["recommended_sdlc"]))

    ranked_options = output.get("ranked_sdlc_options", [])
    st.subheader("Ranked SDLC Fit")
    if isinstance(ranked_options, list) and ranked_options:
        ranked_rows = [
            {
                "Rank": option["rank"],
                "SDLC Model": option["sdlc_model"],
                "Fit": f"{format_fit(option['percentage'])}%",
                "Rationale": option["rationale"],
            }
            for option in ranked_options
            if isinstance(option, dict)
        ]
        if ranked_rows:
            st.caption(
                f"Showing all {len(ranked_rows)} evaluated candidate models."
            )
            st.dataframe(ranked_rows, use_container_width=True, hide_index=True)

    st.subheader("Software Engineering Justification")
    st.write(output["se_justification"])

    st.subheader("Alternative Models Considered")
    alternatives = output.get("alternative_considered", [])
    if isinstance(alternatives, list) and alternatives:
        for alternative in alternatives:
            if isinstance(alternative, dict):
                st.markdown(
                    f"- **{alternative['model']} "
                    f"({format_fit(alternative['percentage'])}% fit)**: "
                    f"{alternative['reason']}"
                )

    st.subheader("Key Risks and Mitigations")
    risks = output.get("key_risks_and_mitigations", [])
    if isinstance(risks, list) and risks:
        for item in risks:
            if isinstance(item, dict):
                with st.container(border=True):
                    st.markdown(f"**Risk:** {item['risk']}")
                    st.markdown(f"**Mitigation:** {item['mitigation']}")

    if STEP2_MARKDOWN.is_file():
        with st.expander("Rendered Markdown", expanded=False):
            st.markdown(STEP2_MARKDOWN.read_text(encoding="utf-8"))
    with st.expander("Raw Step 2 JSON", expanded=False):
        st.json(output)


st.set_page_config(page_title="Banking Requirements & SDLC", layout="wide")
init_state()

st.title("Banking Requirements & SDLC Explorer")

with st.sidebar:
    st.subheader("Banking Reference Files")
    if not DATA_DIR.is_dir():
        st.warning(f"No banking reference directory found at {DATA_DIR}")
    else:
        markdown_files = sorted(DATA_DIR.glob("*.md"))
        if not markdown_files:
            st.warning("No .md reference files found.")
        for path in markdown_files:
            with st.expander(path.name, expanded=False):
                try:
                    st.markdown(path.read_text(encoding="utf-8"))
                except OSError as exc:
                    st.error(f"Unable to read {path.name}: {exc}")

st.subheader("Workflow")
functionality = st.text_input(
    "Target banking functionality",
    placeholder="e.g., Domestic Instant Fund Transfer or Customer Notification and Alert Preferences",
)

step1_col, step2_col = st.columns(2)
with step1_col:
    run_step1 = st.button(
        "Run Step 1 (Generate Requirements)",
        disabled=st.session_state["is_running"],
        type="primary",
    )
with step2_col:
    run_step2 = st.button(
        "Run Step 2 (Select SDLC & Justify)",
        disabled=(
            st.session_state["is_running"]
            or st.session_state["step1_output"] is None
        ),
    )

if run_step1:
    if not functionality.strip():
        st.session_state["error_message"] = (
            "Enter a target banking functionality before running Step 1."
        )
    else:
        st.session_state["pending_step"] = "step1"
        st.session_state["pending_functionality"] = functionality.strip()
        st.session_state["is_running"] = True
        st.rerun()

if run_step2:
    st.session_state["pending_step"] = "step2"
    st.session_state["pending_context"] = None
    st.session_state["is_running"] = True
    st.rerun()

if st.session_state["is_running"] and st.session_state["pending_step"]:
    pending_step = st.session_state["pending_step"]
    try:
        if pending_step == "step1":
            with st.spinner("Generating grounded requirements with the local LLM..."):
                result = generate_requirements(
                    st.session_state["pending_functionality"],
                    data_dir=DATA_DIR,
                    output_path=STEP1_OUTPUT,
                )
            st.session_state["step1_output"] = result
            st.session_state["step2_output"] = None
            st.success(
                f"Step 1 complete: "
                f"{len(result.get('requirements', []))} requirements generated."
            )
        elif pending_step == "step2":
            with st.spinner("Selecting an SDLC and preparing justification..."):
                result = select_sdlc(
                    STEP1_OUTPUT,
                    st.session_state["pending_context"],
                    json_output_path=STEP2_JSON,
                    markdown_output_path=STEP2_MARKDOWN,
                )
            st.session_state["step2_output"] = result
            st.success(
                f"Step 2 complete: recommended "
                f"{result['recommended_sdlc']}."
            )
    except Exception as exc:
        st.session_state["error_message"] = explain_error(exc)
    finally:
        st.session_state["is_running"] = False
        st.session_state["pending_step"] = None
        st.session_state["pending_context"] = None
        st.rerun()

if st.session_state["error_message"]:
    st.error(st.session_state["error_message"])
    st.session_state["error_message"] = None

if st.session_state["step2_output"] is not None:
    st.divider()
    st.header("Step 2 Results")
    render_step2(st.session_state["step2_output"])

if st.session_state["step1_output"] is not None:
    st.divider()
    st.header("Step 1 Results")
    render_step1(st.session_state["step1_output"])
