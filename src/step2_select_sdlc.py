"""Select an SDLC model for generated banking requirements."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS_PATH = PROJECT_ROOT / "outputs" / "step1_requirements.json"
DEFAULT_JSON_PATH = PROJECT_ROOT / "outputs" / "step2_sdlc_justification.json"
DEFAULT_MARKDOWN_PATH = PROJECT_ROOT / "outputs" / "step2_sdlc_justification.md"


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract one JSON object from fences or extra text from a local model."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("The local model returned an empty response instead of JSON.")

    text = raw_text.strip().lstrip("\ufeff")
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    def decode(candidate: str) -> tuple[Any | None, json.JSONDecodeError | None]:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            return None, exc

    # Handle a common local-model artifact: a trailing comma before a closing
    # brace or bracket. It is removed only after JSON parsing fails.
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    parsed, decode_error = decode(repaired)
    if decode_error is not None:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for index, character in enumerate(repaired):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(repaired[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)
        if not candidates:
            preview = " ".join(text.split())
            if len(preview) > 160:
                preview = preview[:157] + "..."
            raise ValueError(
                "The local model returned malformed JSON "
                f"(line {decode_error.lineno}, column {decode_error.colno}). "
                "Return a complete JSON object with all required keys. "
                f"Response preview: {preview or '<empty>'}"
            )
        parsed = max(candidates, key=lambda item: len(json.dumps(item)))

    if not isinstance(parsed, dict):
        raise ValueError("The local model must return a JSON object.")
    return parsed


CANONICAL_KEYS = (
    "recommended_sdlc",
    "alternative_considered",
    "se_justification",
    "key_risks_and_mitigations",
    "ranked_sdlc_options",
)

APPROVED_SDLC_MODELS = {
    "Waterfall",
    "V-Shaped",
    "Prototype",
    "RAD",
    "RAD (Rapid Application Development)",
    "Spiral",
    "Agile",
}

KEY_ALIASES = {
    "recommended_sdlc": (
        "recommended_sdlc", "selected_sdlc", "sdlc_model",
        "recommended_model", "sdlc",
    ),
    "alternative_considered": (
        "alternative_considered", "alternatives_considered", "alternatives",
        "alternative_models",
    ),
    "se_justification": (
        "se_justification", "justification", "engineering_justification",
        "rationale",
    ),
    "key_risks_and_mitigations": (
        "key_risks_and_mitigations", "risks_and_mitigations", "key_risks",
        "risks",
    ),
    "ranked_sdlc_options": (
        "ranked_sdlc_options", "ranked_options", "sdlc_ranking",
        "ranked_models", "model_ranking",
    ),
}


def normalize_sdlc_result(result: dict[str, Any]) -> dict[str, Any]:
    """Unwrap common nested wrappers and canonicalize LLM key aliases."""
    nested_keys = ("sdlc_recommendation", "recommendation", "result", "output")
    for wrapper in nested_keys:
        inner = result.get(wrapper)
        if isinstance(inner, dict):
            normalized_inner = normalize_sdlc_result(inner)
            if any(key in normalized_inner for key in CANONICAL_KEYS):
                result = normalized_inner
                break

    normalized: dict[str, Any] = {}
    lookup = {
        re.sub(r"[^a-z0-9]", "", key.lower()): key
        for key in result
        if isinstance(key, str)
    }
    for canonical, aliases in KEY_ALIASES.items():
        if canonical in result:
            normalized[canonical] = result[canonical]
            continue
        for alias in aliases:
            key = lookup.get(re.sub(r"[^a-z0-9]", "", alias.lower()))
            if key is not None:
                normalized[canonical] = result[key]
                break
    return normalized


def load_requirements(
    path: str | Path = DEFAULT_REQUIREMENTS_PATH,
) -> dict[str, Any]:
    requirements_path = Path(path)
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"Generated requirements not found: {requirements_path}"
        )
    try:
        payload = json.loads(requirements_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Generated requirements contain invalid JSON: {requirements_path}"
        ) from exc

    if isinstance(payload, list):
        requirements = payload
        profile = None
    elif isinstance(payload, dict):
        requirements = payload.get("requirements")
        profile = payload.get("functionality_profile")
        if not isinstance(profile, dict):
            raise ValueError(
                "Step 1 output functionality_profile must be a JSON object."
            )
    else:
        raise ValueError(
            "Step 1 output must be a requirements array or profile-bearing object."
        )

    if not isinstance(requirements, list) or not requirements:
        raise ValueError("Generated requirements must be a non-empty JSON array.")
    for index, item in enumerate(requirements, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Generated requirement {index} must be a JSON object.")
        missing = {"id", "type", "title", "requirement", "reasoning"} - item.keys()
        if missing:
            raise ValueError(
                f"Generated requirement {index} is missing keys: "
                + ", ".join(sorted(missing))
            )
    if profile is not None:
        for key in (
            "target_functionality",
            "estimated_criticality",
            "expected_change_frequency",
        ):
            if not isinstance(profile.get(key), str) or not profile[key].strip():
                raise ValueError(
                    f"functionality_profile.{key} must be a non-empty string."
                )
        if profile["estimated_criticality"] not in {"Low", "Medium", "High"}:
            raise ValueError(
                "functionality_profile estimated_criticality must be Low, Medium, or High."
            )
        if profile["expected_change_frequency"] not in {"Low", "Medium", "High"}:
            raise ValueError(
                "functionality_profile expected_change_frequency must be Low, Medium, or High."
            )
    return {"functionality_profile": profile, "requirements": requirements}


def temperature_from_environment() -> float:
    raw_value = os.getenv("LLM_TEMPERATURE", "0.2")
    try:
        temperature = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"LLM_TEMPERATURE must be a number, received: {raw_value!r}"
        ) from exc
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0")
    return temperature


def build_sdlc_prompt(
    generated_output: dict[str, Any] | list[dict[str, Any]],
    project_context: str | None = None,
) -> str:
    context = project_context.strip() if project_context else ""
    context_text = f"\nOptional project/team context:\n{context}\n" if context else ""
    if isinstance(generated_output, dict):
        profile = generated_output.get("functionality_profile")
        requirements = generated_output["requirements"]
    else:
        profile = None
        requirements = generated_output
    profile_text = json.dumps(profile, indent=2) if profile else "Not provided"
    return f"""Evaluate the exact generated requirements below and recommend one SDLC model.

GROUNDING RULES:
- Use the functionality_profile to distinguish low-risk/preference/UI features from
  high-risk, transactional, or compliance-heavy features.
- For Low criticality or UI/preference work, prefer only a faster/lighter justified
  fit when the requirement set supports it; do not add transaction or compliance
  burden that the requirements do not contain.
- For High criticality or transactional/compliance work, explain why heavier
  verification or risk-driven phases are proportionate to the actual constraints.
- Cite requirement IDs (for example, REQ-F-005 or REQ-NF-003) and quote or name
  their specific constraint in the justification.
- Discuss concrete conflicts and trade-offs from this set: for example, fast
  payment iteration versus immutable auditability, or rapid delivery versus
  AML/security verification and 99.95% availability.
- Explain requirement volatility and clarity from the actual statements.
- Assess security/compliance criticality using the requirements that mention
  authentication, authorization, encryption, audit logs, AML, sanctions, or retention.
- Cover verification needs using the measurable limits, latencies, availability,
  rollback, idempotency, and retention statements; do not invent unlisted targets.
- Treat assumptions in requirement reasoning as uncertainty that needs validation,
  not as proven facts.
- se_justification must explicitly explain whether this feature is better served
  by a lighter/faster model (Agile, Prototype, or RAD) or a heavier verification/
  risk-driven model (V-Shaped, Spiral, or Waterfall), and why that fit is
  proportionate to its profile, constraints, and change frequency.
- Do not give textbook-only model definitions or generic claims such as "Agile
  improves flexibility" without tying them to these requirements.
- A less suitable alternative must explain the honest cost or mismatch for this
  requirement set, not merely state that it is less modern.

MODEL CHOICES AND DECISION CRITERIA:
Choose strictly from Waterfall, V-Shaped, Prototype, RAD (Rapid Application
Development), Spiral, Agile, or a clearly named hybrid of these exact class
models. Compare functionality_profile (criticality and change frequency) plus
requirement clarity, volatility, risk, security/compliance, verification, and
stakeholder cadence against these criteria:

- Waterfall: strong fit for fixed/well-understood requirements, Low change
  frequency, and formal phase sign-offs. Weak fit for evolving requirements or
  UI/UX that needs early stakeholder validation.
- V-Shaped: strong fit for pairwise verification and validation, heavy QA,
  high reliability, and compliance/safety evidence tied to each development
  phase. Weak fit for fast, exploratory work or frequently changing user-facing
  features.
- Prototype: strong fit for unclear requirements or workflows/interfaces needing
  early demonstrations and stakeholder feedback. Weak alone for production-grade
  compliance or security; it needs formal hardening before release.
- RAD (Rapid Application Development): strong fit for fast delivery, reusable components, active user feedback,
  and short build cycles with less upfront documentation. Weak for deep risk
  analysis or strict formal verification as the primary need.
- Spiral: strong fit for high technical, operational, or risk uncertainty
  needing repeated risk analysis, prototyping, evaluation, and controlled
  commitment. Weak for simple, Low-risk features where risk loops add overhead.
- Agile: strong fit for evolving requirements, incremental delivery, frequent
  stakeholder feedback, and modular features. Weak when heavy upfront regulatory
  or verification gates strictly block incremental release.
- Hybrid: allowed only to combine named class models. Each component must map
  explicitly to a concrete requirement or constraint from Step 1, such as
  Prototype for unclear UI rules plus V-Shaped for AML/audit verification.

State why the chosen model fits better than the closest alternatives. For each
alternative_considered entry, compare that model against these exact criteria
and cite specific Step 1 requirement IDs or profile fields. Do not substitute
generic textbook definitions or unsupported benefits.

RANKED OUTPUT RULE:
- ranked_sdlc_options MUST contain at least 3 distinct approved models or
  clearly justified hybrids: Rank 1 is primary and Ranks 2+ are alternatives.
  Use 3 to 6 approved options or clearly justified
  hybrids, sorted strictly by descending percentage.
- Percentages must be strictly decreasing from Rank 1 downward.
- Assign rank starting at 1. Each percentage must be an integer or float from
  0 to 100 based on fit with Step 1 requirements, functionality_profile,
  criticality, change frequency, and concrete risks.
- Each rationale must cite a requirement ID/profile field or explain its direct
  relevance. Scores are independent and do not need to sum to 100.
- recommended_sdlc must exactly match the rank 1 sdlc_model.
- alternative_considered must describe lower-ranked options rather than a
  separate unrelated comparison.

Return exactly one valid JSON object and no Markdown fences or prose:
{{
  "recommended_sdlc": "specific model or explicitly named hybrid",
  "alternative_considered": [
    {{"model": "alternative name", "percentage": 75, "reason": "specific trade-off for these requirements"}},
    {{"model": "alternative name", "percentage": 60, "reason": "specific trade-off for these requirements"}}
  ],
  "se_justification": "thorough paragraph with requirement IDs and explicit trade-offs among iteration speed, security, auditability, verification rigor, and stakeholder/delivery cadence",
  "key_risks_and_mitigations": [
    {{"risk": "risk tied to a listed requirement or assumption", "mitigation": "specific executable mitigation"}}
  ],
  "ranked_sdlc_options": [
    {{"rank": 1, "sdlc_model": "...", "percentage": 90, "rationale": "..."}},
    {{"rank": 2, "sdlc_model": "...", "percentage": 75, "rationale": "..."}},
    {{"rank": 3, "sdlc_model": "...", "percentage": 60, "rationale": "..."}}
  ]
}}

key_risks_and_mitigations is required; do not omit it or rename it.
Provide one or two alternatives and at least three practical risks.
{context_text}
Functionality profile:
{profile_text}

Generated requirements:
{json.dumps(requirements, indent=2, ensure_ascii=False)}

Concise JSON skeleton (top-level keys only):
{{
  "recommended_sdlc": "...",
  "alternative_considered": [{{"model": "...", "percentage": 75, "reason": "..."}}],
  "se_justification": "...",
  "key_risks_and_mitigations": [{{"risk": "...", "mitigation": "..."}}],
  "ranked_sdlc_options": [{{"rank": 1, "sdlc_model": "...", "percentage": 90, "rationale": "..."}}]
}}
"""


def validate_sdlc_result(result: dict[str, Any]) -> dict[str, Any]:
    result = normalize_sdlc_result(result)
    required = {
        "recommended_sdlc",
        "alternative_considered",
        "se_justification",
    }
    missing = required - result.keys()
    if missing:
        for key in sorted(missing):
            if key == "recommended_sdlc":
                result[key] = "Agile"
            elif key == "alternative_considered":
                result[key] = [
                    {
                        "model": "V-Shaped",
                        "reason": "Fallback alternative because the local model omitted alternatives.",
                    }
                ]
            elif key == "se_justification":
                result[key] = (
                    "Fallback justification: choose the SDLC by balancing the "
                    "Step 1 functionality profile, measurable requirements, risk, "
                    "verification needs, and stakeholder/delivery cadence."
                )
        warnings.warn(
            "Local model omitted core fields; added fallbacks for: "
            + ", ".join(sorted(missing)),
            RuntimeWarning,
            stacklevel=2,
        )

    canonical_risk_key = "key_risks_and_mitigations"
    if canonical_risk_key not in result:
        result[canonical_risk_key] = [
            {
                "risk": "Requirements or assumptions may be interpreted inconsistently.",
                "mitigation": "Review requirement IDs and assumptions with stakeholders during backlog refinement.",
            },
            {
                "risk": "Security, compliance, or verification controls may be incomplete.",
                "mitigation": "Map controls and test cases to the applicable Step 1 requirement IDs.",
            },
            {
                "risk": "Delivery may miss availability or latency constraints.",
                "mitigation": "Add measurable load, availability, rollback, and acceptance checks before release.",
            },
        ]
        warnings.warn(
            "Local model omitted key_risks_and_mitigations; "
            "added a sensible fallback risk list.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not isinstance(result["recommended_sdlc"], str) or not result["recommended_sdlc"].strip():
        result["recommended_sdlc"] = "Agile"
        warnings.warn(
            "recommended_sdlc was malformed; used an Agile fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not isinstance(result["se_justification"], str) or not result["se_justification"].strip():
        result["se_justification"] = (
            "Fallback justification: balance the Step 1 profile, requirement "
            "risk, verification needs, change frequency, and stakeholder cadence."
        )
        warnings.warn(
            "se_justification was malformed; used a fallback justification.",
            RuntimeWarning,
            stacklevel=2,
        )

    alternatives = result["alternative_considered"]
    if not isinstance(alternatives, list) or not 1 <= len(alternatives) <= 2:
        alternatives = [
            {
                "model": "V-Shaped",
                "reason": "Fallback alternative because the local model omitted usable alternatives.",
            }
        ]
        warnings.warn(
            "alternative_considered was malformed; used a fallback alternative.",
            RuntimeWarning,
            stacklevel=2,
        )
    normalized_alternatives: list[dict[str, Any]] = []
    for index, alternative in enumerate(alternatives, start=1):
        if not isinstance(alternative, dict):
            continue
        model = alternative.get("model")
        reason = alternative.get("reason")
        if not isinstance(model, str) or not model.strip() or not isinstance(
            reason, str
        ) or not reason.strip():
            continue
        percentage = alternative.get("percentage")
        normalized_alternative = {"model": model.strip(), "reason": reason.strip()}
        if (
            isinstance(percentage, (int, float))
            and not isinstance(percentage, bool)
            and 0 <= float(percentage) <= 100
        ):
            normalized_alternative["percentage"] = percentage
        normalized_alternatives.append(
            normalized_alternative
        )
    if not normalized_alternatives:
        normalized_alternatives = [
            {
                "model": "V-Shaped",
                "reason": "Fallback alternative because no usable alternative was returned.",
            }
        ]
        warnings.warn(
            "Alternative entries were malformed; used a fallback alternative.",
            RuntimeWarning,
            stacklevel=2,
        )

    risks = result["key_risks_and_mitigations"]
    if not isinstance(risks, list) or len(risks) < 3:
        risks = [
            {
                "risk": "Requirements or assumptions may be interpreted inconsistently.",
                "mitigation": "Review requirement IDs and assumptions with stakeholders during backlog refinement.",
            },
            {
                "risk": "Security, compliance, or verification controls may be incomplete.",
                "mitigation": "Map controls and test cases to the applicable Step 1 requirement IDs.",
            },
            {
                "risk": "Delivery may miss availability or latency constraints.",
                "mitigation": "Add measurable load, availability, rollback, and acceptance checks before release.",
            },
        ]
        warnings.warn(
            "key_risks_and_mitigations was malformed or incomplete; "
            "used a fallback risk list.",
            RuntimeWarning,
            stacklevel=2,
        )
    normalized_risks: list[dict[str, str]] = []
    for index, item in enumerate(risks, start=1):
        if not isinstance(item, dict):
            continue
        risk = item.get("risk")
        mitigation = item.get("mitigation")
        if not isinstance(risk, str) or not risk.strip() or not isinstance(
            mitigation, str
        ) or not mitigation.strip():
            continue
        normalized_risks.append(
            {"risk": risk.strip(), "mitigation": mitigation.strip()}
        )
    if len(normalized_risks) < 3:
        normalized_risks = [
            {
                "risk": "Requirements or assumptions may be interpreted inconsistently.",
                "mitigation": "Review requirement IDs and assumptions with stakeholders during backlog refinement.",
            },
            {
                "risk": "Security, compliance, or verification controls may be incomplete.",
                "mitigation": "Map controls and test cases to the applicable Step 1 requirement IDs.",
            },
            {
                "risk": "Delivery may miss availability or latency constraints.",
                "mitigation": "Add measurable load, availability, rollback, and acceptance checks before release.",
            },
        ]

    ranked_raw = result.get("ranked_sdlc_options")
    normalized_ranked: list[dict[str, Any]] = []
    if isinstance(ranked_raw, list):
        for item in ranked_raw:
            if not isinstance(item, dict):
                continue
            model = item.get("sdlc_model")
            percentage = item.get("percentage")
            rationale = item.get("rationale")
            if (
                not isinstance(model, str)
                or not model.strip()
                or isinstance(percentage, bool)
                or not isinstance(percentage, (int, float))
                or not 0 <= float(percentage) <= 100
                or not isinstance(rationale, str)
                or not rationale.strip()
            ):
                continue
            model_name = model.strip()
            is_approved_hybrid = (
                " hybrid" in model_name.lower()
                and any(approved in model_name for approved in APPROVED_SDLC_MODELS)
            )
            if model_name not in APPROVED_SDLC_MODELS and not is_approved_hybrid:
                continue
            normalized_ranked.append(
                {
                    "rank": 0,
                    "sdlc_model": model.strip(),
                    "percentage": percentage,
                    "rationale": rationale.strip(),
                }
            )

    normalized_ranked.sort(
        key=lambda item: float(item["percentage"]),
        reverse=True,
    )

    # Remove duplicate models and enforce strictly decreasing fit percentages.
    distinct_ranked: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    previous_percentage: float | None = None
    for item in normalized_ranked:
        model_name = item["sdlc_model"]
        if model_name in seen_models:
            continue
        percentage = float(item["percentage"])
        if previous_percentage is not None and percentage >= previous_percentage:
            percentage = max(0.0, previous_percentage - 1.0)
        item["percentage"] = int(percentage) if percentage.is_integer() else round(percentage, 2)
        previous_percentage = float(percentage)
        seen_models.add(model_name)
        distinct_ranked.append(item)
    normalized_ranked = distinct_ranked

    for rank, item in enumerate(normalized_ranked, start=1):
        item["rank"] = rank

    if not normalized_ranked:
        normalized_ranked = []

    recommended = result["recommended_sdlc"].strip()
    if normalized_ranked:
        recommended = normalized_ranked[0]["sdlc_model"]
    else:
        recommended = recommended if recommended else "Agile"
        normalized_ranked.append(
            {
                "rank": 1,
                "sdlc_model": recommended,
                "percentage": 75,
                "rationale": "Fallback ranking based on the Step 1 functionality profile.",
            }
        )
        warnings.warn(
            "ranked_sdlc_options was missing or malformed; "
            "added a sensible fallback ranking.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Promote models from alternative_considered when the LLM omitted ranks.
    for alternative in normalized_alternatives:
        model_name = alternative["model"]
        if model_name not in seen_models:
            top_percentage = float(normalized_ranked[0]["percentage"])
            alternative_percentage = alternative.get("percentage")
            if not isinstance(alternative_percentage, (int, float)) or isinstance(
                alternative_percentage, bool
            ):
                alternative_percentage = max(
                    0.0,
                    top_percentage - (len(normalized_ranked) * 10),
                )
            alternative_percentage = float(alternative_percentage)
            normalized_ranked.append(
                {
                    "rank": 0,
                    "sdlc_model": model_name,
                    "percentage": (
                        int(alternative_percentage)
                        if alternative_percentage.is_integer()
                        else round(alternative_percentage, 2)
                    ),
                    "rationale": alternative["reason"],
                }
            )
            seen_models.add(model_name)

    fallback_models = [
        "Agile", "V-Shaped", "Spiral", "RAD", "Prototype", "Waterfall",
    ]
    while len(normalized_ranked) < 3:
        candidate = next(
            (model for model in fallback_models if model not in seen_models),
            None,
        )
        if candidate is None:
            break
        top_percentage = float(normalized_ranked[0]["percentage"])
        fallback_percentage = max(
            0.0,
            top_percentage - (len(normalized_ranked) * 10),
        )
        normalized_ranked.append(
            {
                "rank": 0,
                "sdlc_model": candidate,
                "percentage": (
                    int(fallback_percentage)
                    if fallback_percentage.is_integer()
                    else round(fallback_percentage, 2)
                ),
                "rationale": (
                    f"{candidate} is included as a lower-ranked class alternative "
                    "and should be checked against the concrete Step 1 constraints."
                ),
            }
        )
        seen_models.add(candidate)

    normalized_ranked.sort(
        key=lambda item: float(item["percentage"]),
        reverse=True,
    )
    normalized_ranked = normalized_ranked[:6]
    previous_percentage = None
    for rank, item in enumerate(normalized_ranked, start=1):
        item["rank"] = rank
        percentage = float(item["percentage"])
        if previous_percentage is not None and percentage >= previous_percentage:
            percentage = max(0.0, previous_percentage - 1.0)
            item["percentage"] = int(percentage) if percentage.is_integer() else round(percentage, 2)
        previous_percentage = float(item["percentage"])

    recommended = normalized_ranked[0]["sdlc_model"]
    result["recommended_sdlc"] = recommended

    ranked_alternatives = [
        {
            "model": item["sdlc_model"],
            "percentage": item["percentage"],
            "reason": item["rationale"],
        }
        for item in normalized_ranked[1:]
    ]
    normalized_alternatives = ranked_alternatives

    return {
        "recommended_sdlc": recommended,
        "alternative_considered": normalized_alternatives,
        "se_justification": result["se_justification"].strip(),
        "key_risks_and_mitigations": normalized_risks,
        "ranked_sdlc_options": normalized_ranked,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SDLC Recommendation",
        "",
        f"## Recommended SDLC: {result['recommended_sdlc']}",
        "",
        "## Ranked SDLC Options",
        "",
        "| Rank | SDLC Model | Fit Score | Rationale |",
        "|---:|---|---:|---|",
    ]
    for option in result.get("ranked_sdlc_options", []):
        rationale = str(option["rationale"]).replace("|", "\\|").replace("\n", " ")
        model = str(option["sdlc_model"]).replace("|", "\\|")
        lines.append(
            f"| {option['rank']} | {model} | "
            f"{option['percentage']}% | {rationale} |"
        )

    lines.extend([
        "",
        "## Alternatives Considered",
        "",
    ])
    for alternative in result["alternative_considered"]:
        percentage = alternative.get("percentage")
        fit_text = f" ({percentage}% fit)" if percentage is not None else ""
        lines.append(
            f"- **{alternative['model']}**{fit_text}: {alternative['reason']}"
        )

    lines.extend(["", "## Software Engineering Justification", "", result["se_justification"], "",
                  "## Key Risks and Mitigations", ""])
    for item in result["key_risks_and_mitigations"]:
        lines.append(f"- **Risk:** {item['risk']}")
        lines.append(f"  - **Mitigation:** {item['mitigation']}")
    return "\n".join(lines).rstrip() + "\n"


def write_atomic(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def select_sdlc(
    requirements_path: str | Path = DEFAULT_REQUIREMENTS_PATH,
    project_context: str | None = None,
    *,
    json_output_path: str | Path = DEFAULT_JSON_PATH,
    markdown_output_path: str | Path = DEFAULT_MARKDOWN_PATH,
) -> dict[str, Any]:
    """Evaluate requirements and save/return the structured SDLC recommendation."""
    load_dotenv()
    requirements = load_requirements(requirements_path)
    prompt = build_sdlc_prompt(requirements, project_context)

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1/")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "qwen3:8b")
    temperature = temperature_from_environment()

    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a software engineering lifecycle analyst. "
                        "Use the requirements and return one valid JSON object only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            f"Cannot reach the local LLM at {base_url}. "
            f"Verify the endpoint and that model '{model}' is running."
        ) from exc
    except APIError as exc:
        raise RuntimeError(
            f"The local LLM at {base_url} returned an API error for model "
            f"'{model}': {exc}"
        ) from exc

    if not response.choices:
        raise ValueError("The local model returned no choices.")
    raw_text = response.choices[0].message.content
    if not raw_text:
        raise ValueError("The local model returned an empty response.")

    result = normalize_sdlc_result(extract_json_object(raw_text))
    missing = set(CANONICAL_KEYS) - result.keys()
    if missing:
        warnings.warn(
            "Local model response omitted required fields: "
            + ", ".join(sorted(missing))
            + "; retrying once with a strict JSON request.",
            RuntimeWarning,
            stacklevel=2,
        )
        retry_prompt = (
            "Return strictly valid JSON with exactly these top-level keys: "
            "recommended_sdlc, alternative_considered, se_justification, "
            "key_risks_and_mitigations, and ranked_sdlc_options. "
            "ranked_sdlc_options must be sorted by descending percentage. "
            "Do not use wrappers or aliases. No Markdown or prose."
        )
        try:
            retry = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "Return one valid JSON object with the exact required keys.",
                    },
                    {"role": "user", "content": retry_prompt},
                ],
            )
            retry_text = (
                retry.choices[0].message.content if retry.choices else None
            )
            if retry_text:
                retry_result = normalize_sdlc_result(
                    extract_json_object(retry_text)
                )
                result.update(retry_result)
        except (APIConnectionError, APITimeoutError, APIError, ValueError) as exc:
            warnings.warn(
                f"Strict JSON retry failed ({exc}); using fallback values.",
                RuntimeWarning,
                stacklevel=2,
            )

    result = validate_sdlc_result(result)
    write_atomic(
        json_output_path,
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    )
    write_atomic(markdown_output_path, render_markdown(result))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recommend an SDLC model for generated banking requirements."
    )
    parser.add_argument(
        "requirements_pos",
        nargs="?",
        help="Optional positional path to step 1 requirements JSON.",
    )
    parser.add_argument(
        "context_pos",
        nargs="?",
        help="Optional positional project/team context.",
    )
    parser.add_argument(
        "--requirements",
        default=str(DEFAULT_REQUIREMENTS_PATH),
        help="Path to step 1 requirements JSON.",
    )
    parser.add_argument("--context", default=None, help="Optional project/team context.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    requirements_path = args.requirements_pos or args.requirements
    project_context = args.context_pos or args.context

    try:
        result = select_sdlc(requirements_path, project_context)
    except (FileNotFoundError, ValueError, RuntimeError, APIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Recommended {result['recommended_sdlc']} "
        f"and saved JSON/Markdown outputs under {DEFAULT_JSON_PATH.parent}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())