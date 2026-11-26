"""Select an SDLC model for generated banking requirements."""

from __future__ import annotations

import argparse
import json
import math
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
    "Agile-DevSecOps",
    "Agile + V-Model",
    "Prototype + V-Shaped",
    "Spiral + Agile",
}

# Qualitative class-model criteria, not preassigned scores or percentages.
MODEL_SIGNAL_FIT = {
    "Waterfall": {
        "supports": {
            "stable_scope",
            "change_low",
            "formal_gates",
            "compliance",
            "audit",
        },
        "tradeoffs": {
            "change_high",
            "ui_workflow",
            "uncertainty",
            "fast_delivery",
            "criticality_high",
            "security",
            "availability",
            "transactional_safety",
            "verification",
        },
    },
    "V-Shaped": {
        "supports": {
            "formal_gates",
            "verification",
            "criticality_high",
            "security",
            "compliance",
            "audit",
            "availability",
            "performance",
            "transactional_safety",
        },
        "tradeoffs": {"change_high", "ui_workflow", "uncertainty", "fast_delivery"},
    },
    "Prototype": {
        "supports": {
            "uncertainty",
            "ui_workflow",
            "change_high",
            "stakeholder_feedback",
        },
        "tradeoffs": {
            "criticality_high",
            "security",
            "compliance",
            "audit",
            "formal_gates",
            "verification",
            "transactional_safety",
        },
    },
    "RAD": {
        "supports": {
            "change_high",
            "change_medium",
            "ui_workflow",
            "stakeholder_feedback",
            "fast_delivery",
        },
        "tradeoffs": {
            "criticality_high",
            "security",
            "compliance",
            "audit",
            "formal_gates",
            "verification",
            "transactional_safety",
        },
    },
    "Spiral": {
        "supports": {
            "criticality_high",
            "security",
            "compliance",
            "audit",
            "availability",
            "performance",
            "transactional_safety",
            "uncertainty",
            "assumption",
            "verification",
            "formal_gates",
        },
        "tradeoffs": {"criticality_low", "change_low", "fast_delivery"},
    },
    "Agile": {
        "supports": {
            "change_high",
            "change_medium",
            "ui_workflow",
            "stakeholder_feedback",
            "modular_delivery",
        },
        "tradeoffs": {
            "formal_gates",
            "verification",
            "compliance",
            "audit",
            "security",
        },
    },
    "Agile-DevSecOps": {
        "supports": {
            "change_high",
            "change_medium",
            "security",
            "compliance",
            "audit",
            "availability",
            "performance",
            "transactional_safety",
            "modular_delivery",
        },
        "tradeoffs": {"stable_scope"},
    },
}

HYBRID_COMPONENTS = {
    "Agile + V-Model": ("Agile", "V-Shaped"),
    "Prototype + V-Shaped": ("Prototype", "V-Shaped"),
    "Spiral + Agile": ("Spiral", "Agile"),
    "Agile-DevSecOps": ("Agile", "V-Shaped"),
}

SCORING_CANDIDATES = tuple(dict.fromkeys((*MODEL_SIGNAL_FIT, *HYBRID_COMPONENTS)))

KEY_ALIASES = {
    "recommended_sdlc": (
        "recommended_sdlc",
        "selected_sdlc",
        "sdlc_model",
        "recommended_model",
        "sdlc",
    ),
    "alternative_considered": (
        "alternative_considered",
        "alternatives_considered",
        "alternatives",
        "alternative_models",
    ),
    "se_justification": (
        "se_justification",
        "justification",
        "engineering_justification",
        "rationale",
    ),
    "key_risks_and_mitigations": (
        "key_risks_and_mitigations",
        "risks_and_mitigations",
        "key_risks",
        "risks",
    ),
    "ranked_sdlc_options": (
        "ranked_sdlc_options",
        "ranked_options",
        "sdlc_ranking",
        "ranked_models",
        "model_ranking",
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


def sdlc_response_issues(
    result: dict[str, Any],
    generated_output: dict[str, Any],
    project_context: str | None,
) -> list[str]:
    """Identify schema and grounding defects worth one strict-response retry."""
    factors = analyze_step1_factors(generated_output, project_context)
    issues: list[str] = []
    ranking = result.get("ranked_sdlc_options")
    if not isinstance(ranking, list) or len(ranking) < 3:
        issues.append("ranked_sdlc_options needs at least three entries")
    else:
        seen_models: set[str] = set()
        percentages: list[float] = []
        for item in ranking:
            if not isinstance(item, dict):
                issues.append("a ranked option is not an object")
                continue
            model = normalize_model_name(item.get("sdlc_model", item.get("model")))
            if not model or model in seen_models:
                issues.append("ranked options contain a missing or duplicate model")
                continue
            seen_models.add(model)
            percentage = item.get("percentage")
            if (
                not isinstance(percentage, (int, float))
                or isinstance(percentage, bool)
                or not math.isfinite(float(percentage))
                or not 0 <= float(percentage) <= 100
            ):
                issues.append(f"{model} has an invalid percentage")
                continue
            percentages.append(float(percentage))
            rationale = item.get("rationale")
            if not isinstance(rationale, str) or not rationale_matches_step1(
                rationale, factors
            ):
                issues.append(f"{model} rationale lacks current requirement evidence")
        if any(left < right for left, right in zip(percentages, percentages[1:])):
            issues.append("ranked percentages are not sorted highest to lowest")
        missing_models = set(SCORING_CANDIDATES) - seen_models
        if missing_models:
            issues.append(
                "ranked options omit evaluated candidates: "
                + ", ".join(sorted(missing_models))
            )
        first_model = (
            normalize_model_name(ranking[0].get("sdlc_model"))
            if isinstance(ranking[0], dict)
            else None
        )
        if normalize_model_name(result.get("recommended_sdlc")) != first_model:
            issues.append("recommended_sdlc does not match rank one")

    justification = result.get("se_justification")
    if not isinstance(justification, str) or not rationale_matches_step1(
        justification, factors
    ):
        issues.append("se_justification lacks current requirement evidence")

    alternatives = result.get("alternative_considered")
    if not isinstance(alternatives, list) or not alternatives:
        issues.append("alternative_considered is missing lower-ranked models")
    else:
        for item in alternatives:
            if not isinstance(item, dict):
                issues.append("an alternative entry is not an object")
                continue
            model = normalize_model_name(item.get("model", item.get("sdlc_model")))
            reason = item.get("reason", item.get("rationale"))
            percentage = item.get("percentage")
            if (
                not model
                or not isinstance(reason, str)
                or not rationale_matches_step1(reason, factors)
                or not isinstance(percentage, (int, float))
                or isinstance(percentage, bool)
                or not math.isfinite(float(percentage))
                or not 0 <= float(percentage) <= 100
            ):
                issues.append(
                    "an alternative entry lacks a grounded model, score, or reason"
                )
    risks = result.get("key_risks_and_mitigations")
    if not isinstance(risks, list) or len(risks) < 3:
        issues.append("key_risks_and_mitigations needs three current-run risks")
    else:
        grounded_risks = [
            item
            for item in risks
            if isinstance(item, dict)
            and isinstance(item.get("risk"), str)
            and isinstance(item.get("mitigation"), str)
            and rationale_matches_step1(f"{item['risk']} {item['mitigation']}", factors)
        ]
        if len(grounded_risks) < 3:
            issues.append("risk mitigations lack current requirement evidence")
    return issues


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
        generated_output = {"functionality_profile": None, "requirements": requirements}
    profile_text = json.dumps(profile, indent=2) if profile else "Not provided"
    factor_signals = analyze_step1_factors(generated_output, context)
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
  transaction iteration versus immutable auditability, or rapid delivery versus
  the security and availability constraints actually stated below.
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
Choose from Waterfall, V-Shaped (also called V-Model), Prototype, RAD (Rapid
Application Development), Spiral, Agile, or a clearly named hybrid of these
class models. Agile-DevSecOps and V-Model hybrids are allowed only when each
component is justified by current requirement IDs. Compare functionality_profile
(criticality and change frequency), requirement clarity, volatility, risk,
security/compliance, verification, operational demands, and stakeholder cadence:

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
- ranked_sdlc_options MUST contain every distinct evaluated candidate model
  listed below: Rank 1 is primary and all remaining ranks are alternatives.
  Sort highest to lowest. Scores may be close when the fit is close, but do not
  describe distinct displayed scores as tied.
- Assign rank starting at 1. Each percentage must be an integer or float from
  0 to 100 based on fit with Step 1 requirements, functionality_profile,
  criticality, change frequency, and concrete risks.
- Assess each candidate using the supplied current-run factor signals:
  criticality, change frequency, counts of security/compliance/audit/
  availability/performance/operational requirements, and assumption flags.
  Treat these as measurable inputs, not generic banking defaults.
- Explain the score using these signals; do not copy any percentage from the
  schema or invent limits not present in the requirements.
- A score is the current-run factor score refined by the amount of relevant
  Step 1 evidence cited in the rationale. Explain which requirement IDs change
  the model's fit; Python blends the model score with the signal-derived score
  in proportion to the fraction of relevant requirement IDs cited.
- Keep each rationale focused on evidence and trade-offs, not an exact numeric
  score: Python may lower a rounded tie by 0.01 to display a strictly descending
  ranking while preserving the original score and tie-break order. Any generated
  fallback rationale that states a score uses the final displayed percentage.
- Let score differences emerge from the current-run factor and evidence signals.
  Python preserves the score sort/tie-break order and adjusts any rounded
  percentage that would equal or exceed the preceding displayed percentage
  downward by 0.01 (not below 0.0). Ensure each rationale explains the model's
  actual relative fit from current-run evidence, not an artificial score gap.
- Every ranked item, including Ranks 2 and 3, must have a specific rationale
  tied to at least one Step 1 requirement ID or a functionality_profile field
  (criticality, change frequency, or target functionality). Generic statements
  about model benefits alone are not acceptable.
- Each rationale must cite a requirement ID/profile field or explain its direct
  relevance. Scores are independent and do not need to sum to 100.
- recommended_sdlc must exactly match the rank 1 sdlc_model.
- alternative_considered must describe lower-ranked options rather than a
  separate unrelated comparison.

Return exactly one valid JSON object and no Markdown fences or prose:
{{
  "recommended_sdlc": "specific model or explicitly named hybrid",
  "alternative_considered": [],
  "se_justification": "thorough paragraph with requirement IDs and explicit trade-offs among iteration speed, security, auditability, verification rigor, and stakeholder/delivery cadence",
  "key_risks_and_mitigations": [],
  "ranked_sdlc_options": []
}}

key_risks_and_mitigations is required; do not omit it or rename it.
Provide every evaluated candidate and practical risks grounded in the current
requirements. `alternative_considered` must describe all lower-ranked options.
Evaluate all candidates: {json.dumps(SCORING_CANDIDATES)}
{context_text}
Functionality profile:
{profile_text}

Requirement-derived factor signals (use these to score this run):
{json.dumps(factor_signals, indent=2, ensure_ascii=False)}

Generated requirements:
{json.dumps(requirements, indent=2, ensure_ascii=False)}

Concise JSON skeleton (top-level keys only):
{{
  "recommended_sdlc": "...",
  "alternative_considered": [],
  "se_justification": "...",
  "key_risks_and_mitigations": [],
  "ranked_sdlc_options": []
}}
"""


def has_specific_ranking_rationale(rationale: str) -> bool:
    """Keep only rationale text that names evidence from this generation."""
    return bool(rationale.strip())


SIGNAL_PATTERNS = {
    "security": r"\b(mfa|authentication|authorization|encrypt\w*|tls|security|access control|credential)\b",
    "compliance": r"\b(aml|sanction\w*|pep|compliance|regulatory|kyc)\b",
    "audit": r"\b(audit\w*|retention|immutable|ledger)\b",
    "availability": r"\b(availability|uptime|failover|disaster recovery)\b",
    "performance": r"\b(latency|response time|throughput|performance|load test)\b",
    "operations": r"\b(error handling|rollback|recovery|idempotency|atomic|transaction|backup|retry)\b",
    "ui_workflow": r"\b(ui|user interface|preference|notification|alert|workflow|screen)\b",
    "transactional_safety": r"\b(transfer|debit|credit|payment|transaction|idempotency|atomic|rollback)\b",
    "modular_delivery": r"\b(modular|component|service|api|incremental)\b",
    "uncertainty": r"\b(unclear|uncertain|tbd|validate with|discover|explore)\b",
    "stakeholder_feedback": r"\b(stakeholder feedback|user feedback|customer feedback|demonstration|user testing)\b",
    "formal_gates": r"\b(formal sign.?off|phase gate|verification|validation|regulatory gate)\b",
    "verification": r"\b(verification|validation|acceptance test|qa|test coverage|traceability matrix)\b",
}

SIGNAL_LABELS = {
    "security": "security and authentication controls",
    "compliance": "compliance screening and regulatory obligations",
    "audit": "auditability and retention controls",
    "availability": "availability and recovery constraints",
    "performance": "performance and latency targets",
    "operations": "operational error, rollback, and recovery behavior",
    "ui_workflow": "user-facing workflow and preference behavior",
    "transactional_safety": "transaction integrity and payment behavior",
    "modular_delivery": "modular or incremental delivery",
    "uncertainty": "unclear requirements or explicit assumptions",
    "stakeholder_feedback": "stakeholder or user feedback needs",
    "formal_gates": "formal verification and sign-off requirements",
    "verification": "test, verification, and acceptance evidence",
    "criticality_high": "High estimated criticality",
    "criticality_low": "Low estimated criticality",
    "change_high": "High expected change frequency",
    "change_medium": "Medium expected change frequency",
    "change_low": "Low expected change frequency",
    "stable_scope": "fixed or stable scope",
    "fast_delivery": "a time-constrained delivery objective",
}


def _normalized_profile_level(value: Any) -> str | None:
    if isinstance(value, str) and value.title() in {"Low", "Medium", "High"}:
        return value.title()
    return None


def _model_signal_profile(model: str) -> tuple[set[str], set[str]]:
    if model in MODEL_SIGNAL_FIT:
        traits = MODEL_SIGNAL_FIT[model]
        return set(traits["supports"]), set(traits["tradeoffs"])
    components = HYBRID_COMPONENTS.get(model)
    if not components:
        return set(), set()
    component_profiles = [_model_signal_profile(component) for component in components]
    supports = set().union(*(profile[0] for profile in component_profiles))
    tradeoffs = (
        set.intersection(*(profile[1] for profile in component_profiles)) - supports
    )
    return supports, tradeoffs


def normalize_model_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    model = re.sub(r"\s+", " ", value.strip())
    aliases = {
        "v-model": "V-Shaped",
        "v model": "V-Shaped",
        "v shaped": "V-Shaped",
        "agile + v-shaped": "Agile + V-Model",
        "v-shaped + agile": "Agile + V-Model",
        "v-model + agile": "Agile + V-Model",
        "rad rapid application development": "RAD",
        "rad (rapid application development)": "RAD",
        "agile devsecops": "Agile-DevSecOps",
    }
    alias = aliases.get(model.casefold())
    if alias:
        return alias
    return next(
        (
            candidate
            for candidate in APPROVED_SDLC_MODELS
            if model.casefold() == candidate.casefold()
        ),
        None,
    )


def analyze_step1_factors(
    generated_output: dict[str, Any] | None,
    project_context: str | None = None,
) -> dict[str, Any]:
    """Extract evidence counts and profile/context signals from the current run."""
    generated_output = generated_output or {}
    profile = generated_output.get("functionality_profile") or {}
    requirements = generated_output.get("requirements") or []
    category_ids = {name: [] for name in SIGNAL_PATTERNS}
    assumption_ids: list[str] = []
    requirement_records: list[dict[str, str]] = []

    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        requirement_id = str(requirement.get("id", "")).strip()
        title = str(requirement.get("title", "")).strip()
        statement = str(requirement.get("requirement", "")).strip()
        reasoning = str(requirement.get("reasoning", "")).strip()
        text = f"{title} {statement}".casefold()
        requirement_records.append(
            {
                "id": requirement_id,
                "type": str(requirement.get("type", "")),
                "title": title,
                "requirement": statement,
            }
        )
        for category, pattern in SIGNAL_PATTERNS.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                category_ids[category].append(requirement_id)
        if "[assumption]" in reasoning.casefold():
            assumption_ids.append(requirement_id)

    target = profile.get("target_functionality")
    if not isinstance(target, str) or not target.strip():
        target = next(
            (item["title"] for item in requirement_records if item["title"]),
            "Banking functionality",
        )
    target = target.strip()
    context = project_context.strip() if project_context else ""
    context_patterns = {
        "stable_scope": r"\b(fixed scope|stable requirements|scope is fixed|no planned changes)\b",
        "fast_delivery": r"\b(tight deadline|early delivery|rapid delivery|short deadline)\b",
        "stakeholder_feedback": r"\b(stakeholder feedback|user feedback|customer feedback|frequent demos)\b",
        "formal_gates": r"\b(formal sign.?off|regulatory gate|release gate)\b",
    }
    context_signals = {
        signal: len(re.findall(pattern, context, flags=re.IGNORECASE))
        for signal, pattern in context_patterns.items()
    }

    target_text = target.casefold()
    target_is_ui_oriented = bool(
        re.search(SIGNAL_PATTERNS["ui_workflow"], target_text, re.IGNORECASE)
    )

    criticality = _normalized_profile_level(profile.get("estimated_criticality"))
    if criticality is None:
        has_controls = any(
            category_ids[key] for key in ("security", "compliance", "audit")
        )
        has_transactions = bool(category_ids["transactional_safety"])
        if category_ids["compliance"] or (has_controls and has_transactions):
            criticality = "High"
        elif category_ids["ui_workflow"] and not has_controls:
            criticality = "Low"
        else:
            criticality = "Medium"

    change_frequency = _normalized_profile_level(
        profile.get("expected_change_frequency")
    )
    if change_frequency is None:
        if category_ids["ui_workflow"] or re.search(
            r"\b(evolving|frequent changes|iterative feedback)\b",
            context,
            flags=re.IGNORECASE,
        ):
            change_frequency = "High"
        elif context_signals["stable_scope"]:
            change_frequency = "Low"
        else:
            change_frequency = "Medium"

    signal_requirement_ids = {
        key: sorted(set(ids)) for key, ids in category_ids.items()
    }
    signal_weights = {key: len(set(ids)) for key, ids in signal_requirement_ids.items()}
    signal_weights["ui_workflow"] += int(target_is_ui_oriented)
    signal_weights.update(
        {
            "criticality_high": int(criticality == "High"),
            "criticality_low": int(criticality == "Low"),
            "change_high": int(change_frequency == "High"),
            "change_medium": int(change_frequency == "Medium"),
            "change_low": int(change_frequency == "Low"),
            "assumption": len(set(assumption_ids)),
            "uncertainty": signal_weights["uncertainty"],
            "stable_scope": context_signals["stable_scope"]
            + int(change_frequency == "Low"),
            "fast_delivery": context_signals["fast_delivery"],
            "stakeholder_feedback": signal_weights["stakeholder_feedback"]
            + context_signals["stakeholder_feedback"],
            "formal_gates": signal_weights["formal_gates"]
            + context_signals["formal_gates"],
        }
    )
    signal_weights = {key: value for key, value in signal_weights.items() if value > 0}
    return {
        "target": target,
        "criticality": criticality,
        "change_frequency": change_frequency,
        "context": context,
        "context_signals": context_signals,
        "signal_weights": signal_weights,
        "signal_requirement_ids": signal_requirement_ids,
        "category_counts": {
            key: len(ids) for key, ids in signal_requirement_ids.items()
        },
        "assumption_ids": sorted(set(assumption_ids)),
        "requirements": requirement_records,
        "requirement_ids": [item["id"] for item in requirement_records if item["id"]],
    }


def score_sdlc_models(
    factors: dict[str, Any],
    candidates: Iterable[str] = SCORING_CANDIDATES,
) -> dict[str, float]:
    """Score current signal coverage, explicit mismatches, and hybrid overhead."""
    signal_weights = factors["signal_weights"]
    total_demand = sum(signal_weights.values())
    if total_demand == 0:
        raise ValueError(
            "Step 1 requirements and functionality profile produced no scoreable signals."
        )

    scores: dict[str, float] = {}
    for model in candidates:
        normalized_model = normalize_model_name(model)
        if normalized_model is None:
            continue
        supports, tradeoffs = _model_signal_profile(normalized_model)
        supported_demand = sum(
            weight for signal, weight in signal_weights.items() if signal in supports
        )
        conflicting_demand = sum(
            weight for signal, weight in signal_weights.items() if signal in tradeoffs
        )
        components = HYBRID_COMPONENTS.get(normalized_model, (normalized_model,))
        coordination_demand = (
            (len(components) - 1)
            * total_demand
            / max(1, len(factors["requirement_ids"]))
        )
        evaluated_demand = total_demand + conflicting_demand + coordination_demand
        scores[normalized_model] = (
            100.0 * supported_demand / evaluated_demand if evaluated_demand else 0.0
        )
    return scores


def rationale_matches_step1(rationale: str, factors: dict[str, Any]) -> bool:
    text = rationale.casefold()
    matched_requirements = [
        item
        for item in factors["requirements"]
        if item["id"] and item["id"].casefold() in text
    ]
    if not matched_requirements:
        return False
    if factors["target"].casefold() in text:
        return True
    if ("criticality" in text and factors["criticality"].casefold() in text) or (
        ("change frequency" in text or "change_frequency" in text)
        and factors["change_frequency"].casefold() in text
    ):
        return True
    stop_words = {
        "about",
        "after",
        "before",
        "between",
        "customer",
        "from",
        "must",
        "requirement",
        "shall",
        "should",
        "that",
        "their",
        "this",
        "with",
    }

    def normalize_term(term: str) -> str:
        if term.endswith("ies"):
            return term[:-3] + "y"
        if term.endswith("s") and len(term) > 4 and not term.endswith("ss"):
            return term[:-1]
        return term

    rationale_words = {
        normalize_term(word) for word in re.findall(r"\b[a-z0-9-]{4,}\b", text)
    }
    for requirement in matched_requirements:
        terms = {
            normalize_term(word)
            for word in re.findall(
                r"\b[a-z0-9-]{4,}\b",
                f"{requirement['title']} {requirement['requirement']}".casefold(),
            )
        } - stop_words
        if rationale_words & terms:
            return True
    return False


def model_rationale(
    model: str,
    factors: dict[str, Any],
    percentage: float,
) -> str:
    supports, tradeoffs = _model_signal_profile(model)
    weights = factors["signal_weights"]
    matched = sorted(signal for signal in supports if weights.get(signal, 0) > 0)
    mismatched = sorted(signal for signal in tradeoffs if weights.get(signal, 0) > 0)
    evidence_ids = sorted(
        {
            requirement_id
            for signal in matched + mismatched
            for requirement_id in factors["signal_requirement_ids"].get(signal, [])
        }
    )
    requirement_by_id = {
        item["id"]: item for item in factors["requirements"] if item["id"]
    }
    cited_ids = evidence_ids or factors["requirement_ids"]
    evidence = ", ".join(
        (
            f"{requirement_id} ({requirement_by_id[requirement_id]['title']})"
            if requirement_id in requirement_by_id
            else requirement_id
        )
        for requirement_id in cited_ids
    )
    matched_text = ", ".join(SIGNAL_LABELS.get(signal, signal) for signal in matched)
    if not matched_text:
        matched_text = (
            f"profile criticality={factors['criticality']} and "
            f"change frequency={factors['change_frequency']}"
        )
    tradeoff_text = ", ".join(
        SIGNAL_LABELS.get(signal, signal) for signal in mismatched
    )
    rationale = (
        f"{model} scores {percentage:.2f} for {factors['target']} "
        f"({factors['criticality']} criticality; "
        f"{factors['change_frequency']} expected change frequency)."
    )
    if matched:
        rationale += f" Its fit in {matched_text} is supported by {evidence}."
    else:
        rationale += f" This fit follows the current profile signals and {evidence}."
    if tradeoff_text:
        rationale += f" Its trade-offs are {tradeoff_text}, also present in {evidence}."
    if factors["assumption_ids"]:
        rationale += f" Validate assumptions in {', '.join(factors['assumption_ids'])}."
    return rationale


def dynamic_justification(
    ranking: list[dict[str, Any]],
    factors: dict[str, Any],
) -> str:
    first = ranking[0]
    second = ranking[1]
    relevant_ids = factors["requirement_ids"]
    evidence = ", ".join(relevant_ids)
    counts = ", ".join(
        f"{category}={count}"
        for category, count in factors["category_counts"].items()
        if count
    )
    signal_summary = counts or ", ".join(
        f"{item['id']} ({item['title']})" for item in factors["requirements"]
    )
    context_summary = (
        f" Context notes: {factors['context']}." if factors["context"] else ""
    )
    return (
        f"{first['sdlc_model']} ranks first at {first['percentage']}% for "
        f"{factors['target']} ({factors['criticality']} criticality; "
        f"{factors['change_frequency']} expected change frequency). The current "
        f"requirements {evidence} yield {signal_summary}; the closest alternative "
        f"is {second['sdlc_model']} at {second['percentage']}%, whose trade-off is "
        f"described in its ranked rationale.{context_summary}"
    )


def dynamic_risks(factors: dict[str, Any]) -> list[dict[str, str]]:
    requirement_by_id = {
        item["id"]: item for item in factors["requirements"] if item["id"]
    }
    risks: list[dict[str, str]] = []

    def add_risk(requirement_id: str, concern: str, action: str) -> None:
        item = requirement_by_id.get(requirement_id)
        if not item:
            return
        risks.append(
            {
                "risk": (
                    f"{concern} may be missed for {requirement_id} "
                    f"({item['title']})."
                ),
                "mitigation": f"Trace {requirement_id} to {action} before release.",
            }
        )

    for requirement_id in factors["assumption_ids"]:
        add_risk(
            requirement_id,
            "An unverified assumption",
            "stakeholder confirmation and a testable acceptance criterion",
        )
    for category, concern, action in (
        ("security", "A security control", "an automated security test"),
        ("compliance", "A compliance control", "a compliance evidence check"),
        ("audit", "An audit or retention obligation", "an auditability test"),
        ("availability", "An availability target", "a resilience test"),
        ("performance", "A performance constraint", "a measurable load test"),
        ("operations", "Operational failure behavior", "a failure and recovery test"),
    ):
        for requirement_id in factors["signal_requirement_ids"].get(category, []):
            add_risk(requirement_id, concern, action)
    for requirement_id in factors["requirement_ids"]:
        if len(risks) >= 3:
            break
        add_risk(
            requirement_id,
            "A generated acceptance condition",
            "a requirement-linked acceptance test",
        )
    return risks[: max(3, min(len(risks), len(factors["requirement_ids"])))]


def rationale_evidence_coverage(
    model: str,
    rationale: str,
    factors: dict[str, Any],
) -> float:
    """Measure how much of a model's relevant Step 1 evidence the LLM cited."""
    supports, tradeoffs = _model_signal_profile(model)
    relevant_ids = {
        requirement_id
        for signal in supports | tradeoffs
        for requirement_id in factors["signal_requirement_ids"].get(signal, [])
    }
    if not relevant_ids:
        relevant_ids = set(factors["requirement_ids"])
    cited_ids = {
        requirement_id
        for requirement_id in relevant_ids
        if requirement_id.casefold() in rationale.casefold()
    }
    if not relevant_ids:
        return 0.0
    return len(cited_ids) / len(relevant_ids)


def validate_sdlc_result(
    result: dict[str, Any],
    generated_output: dict[str, Any] | None = None,
    project_context: str | None = None,
) -> dict[str, Any]:
    result = normalize_sdlc_result(result)
    missing_fields = set(CANONICAL_KEYS) - result.keys()
    if missing_fields:
        warnings.warn(
            "Step 2 response still lacks usable fields after parsing: "
            + ", ".join(sorted(missing_fields))
            + "; deriving the affected output from current-run requirements.",
            RuntimeWarning,
            stacklevel=2,
        )
    factors = analyze_step1_factors(generated_output, project_context)
    derived_scores = score_sdlc_models(factors)
    supplied_scores: dict[str, float] = {}
    supplied_rationales: dict[str, str] = {}
    candidate_order: list[str] = []

    def remember_candidate(name: Any) -> str | None:
        normalized = normalize_model_name(name)
        if normalized and normalized not in candidate_order:
            candidate_order.append(normalized)
        return normalized

    remember_candidate(result.get("recommended_sdlc"))
    raw_ranked = result.get("ranked_sdlc_options")
    if isinstance(raw_ranked, list):
        for item in raw_ranked:
            if not isinstance(item, dict):
                continue
            model = remember_candidate(item.get("sdlc_model", item.get("model")))
            if not model:
                continue
            percentage = item.get(
                "percentage", item.get("fit_score", item.get("suitability"))
            )
            if (
                isinstance(percentage, (int, float))
                and not isinstance(percentage, bool)
                and math.isfinite(float(percentage))
                and 0 <= float(percentage) <= 100
            ):
                supplied_scores[model] = float(percentage)
            rationale = item.get("rationale", item.get("reason"))
            if (
                isinstance(rationale, str)
                and has_specific_ranking_rationale(rationale)
                and rationale_matches_step1(rationale, factors)
            ):
                supplied_rationales[model] = rationale.strip()

    alternatives = result.get("alternative_considered")
    if isinstance(alternatives, list):
        for item in alternatives:
            if not isinstance(item, dict):
                continue
            model = remember_candidate(item.get("model", item.get("sdlc_model")))
            if not model:
                continue
            percentage = item.get("percentage")
            if (
                isinstance(percentage, (int, float))
                and not isinstance(percentage, bool)
                and math.isfinite(float(percentage))
                and 0 <= float(percentage) <= 100
            ):
                supplied_scores.setdefault(model, float(percentage))
            reason = item.get("reason", item.get("rationale"))
            if isinstance(reason, str) and rationale_matches_step1(reason, factors):
                supplied_rationales.setdefault(model, reason.strip())

    derived_order = sorted(derived_scores, key=derived_scores.__getitem__, reverse=True)
    for model in derived_order:
        if model not in candidate_order:
            candidate_order.append(model)

    ranked_scores = [
        (
            model,
            (
                derived_scores[model]
                + rationale_evidence_coverage(
                    model,
                    supplied_rationales[model],
                    factors,
                )
                * (supplied_scores[model] - derived_scores[model])
                if model in supplied_scores and model in supplied_rationales
                else derived_scores[model]
            ),
            derived_scores[model],
        )
        for model in candidate_order
    ]
    ranked_scores.sort(key=lambda item: (item[1], item[2]), reverse=True)

    displayed_scores: list[tuple[str, float]] = []
    previous_percentage: float | None = None
    for model, score, _ in ranked_scores:
        percentage = round(score, 2)
        if previous_percentage is not None and percentage >= previous_percentage:
            percentage = max(0.0, round(previous_percentage - 0.01, 2))
        displayed_scores.append((model, percentage))
        previous_percentage = percentage

    ranked_options: list[dict[str, Any]] = []
    dynamic_score_models: list[str] = []
    dynamic_rationale_models: list[str] = []
    for model, percentage in displayed_scores:
        if model not in supplied_scores or model not in supplied_rationales:
            dynamic_score_models.append(model)
        rationale = supplied_rationales.get(model)
        if not rationale:
            rationale = model_rationale(model, factors, percentage)
            dynamic_rationale_models.append(model)
        ranked_options.append(
            {
                "rank": len(ranked_options) + 1,
                "sdlc_model": model,
                "percentage": percentage,
                "rationale": rationale,
            }
        )

    if len(ranked_options) < 3:
        raise ValueError(
            "The current Step 1 output does not provide enough distinct SDLC "
            "candidates for a grounded ranking."
        )

    recommended = ranked_options[0]["sdlc_model"]
    justification = result.get("se_justification")
    if not isinstance(justification, str) or not rationale_matches_step1(
        justification, factors
    ):
        justification = dynamic_justification(ranked_options, factors)

    dynamic_risk_options = dynamic_risks(factors)
    raw_risks = result.get("key_risks_and_mitigations")
    normalized_risks: list[dict[str, str]] = []
    if isinstance(raw_risks, list):
        for item in raw_risks:
            if not isinstance(item, dict):
                continue
            risk = item.get("risk")
            mitigation = item.get("mitigation")
            if (
                isinstance(risk, str)
                and risk.strip()
                and isinstance(mitigation, str)
                and mitigation.strip()
                and rationale_matches_step1(f"{risk} {mitigation}", factors)
            ):
                normalized_risks.append(
                    {"risk": risk.strip(), "mitigation": mitigation.strip()}
                )
    for item in dynamic_risk_options:
        if len(normalized_risks) >= 3:
            break
        if item not in normalized_risks:
            normalized_risks.append(item)

    derived_fields = []
    if dynamic_score_models:
        derived_fields.append(
            "scores for " + ", ".join(dict.fromkeys(dynamic_score_models))
        )
    if dynamic_rationale_models:
        derived_fields.append("rationales for " + ", ".join(dynamic_rationale_models))
    if not isinstance(
        result.get("se_justification"), str
    ) or not rationale_matches_step1(str(result.get("se_justification", "")), factors):
        derived_fields.append("the engineering justification")
    if len(normalized_risks) < 3:
        derived_fields.append("requirement-linked risk mitigations")
    if derived_fields:
        warnings.warn(
            "Derived " + "; ".join(derived_fields) + " from current-run evidence.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "recommended_sdlc": recommended,
        "alternative_considered": [
            {
                "model": item["sdlc_model"],
                "percentage": item["percentage"],
                "reason": item["rationale"],
            }
            for item in ranked_options[1:]
        ],
        "se_justification": justification.strip(),
        "key_risks_and_mitigations": normalized_risks,
        "ranked_sdlc_options": ranked_options,
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

    lines.extend(
        [
            "",
            "## Alternatives Considered",
            "",
        ]
    )
    for alternative in result["alternative_considered"]:
        percentage = alternative.get("percentage")
        fit_text = f" ({percentage}% fit)" if percentage is not None else ""
        lines.append(f"- **{alternative['model']}**{fit_text}: {alternative['reason']}")

    lines.extend(
        [
            "",
            "## Software Engineering Justification",
            "",
            result["se_justification"],
            "",
            "## Key Risks and Mitigations",
            "",
        ]
    )
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

    parse_error: ValueError | None = None
    try:
        result = normalize_sdlc_result(extract_json_object(raw_text))
    except ValueError as exc:
        result = {}
        parse_error = exc
    missing = set(CANONICAL_KEYS) - result.keys()
    response_issues = (
        sdlc_response_issues(result, requirements, project_context)
        if not parse_error
        else ["response was not valid JSON"]
    )
    if missing or response_issues or parse_error:
        warnings.warn(
            "The local model response was incomplete, malformed, or ungrounded; "
            "retrying once with the current requirements and factor signals. "
            + "; ".join(response_issues),
            RuntimeWarning,
            stacklevel=2,
        )
        retry_prompt = f"""{prompt}

RESPONSE CORRECTION:
Return one valid JSON object with the exact top-level keys shown above.
Include every evaluated candidate in the ranking. Each percentage must be
derived from the factor signals and be in the permitted range. Every rationale must
refer to an actual requirement ID or the current functionality_profile.
Do not use wrappers, aliases, Markdown, prose, or placeholder values.
Correct these validation issues: {json.dumps(response_issues)}
"""
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
            retry_text = retry.choices[0].message.content if retry.choices else None
            if retry_text:
                retry_result = normalize_sdlc_result(extract_json_object(retry_text))
                result.update(retry_result)
        except (APIConnectionError, APITimeoutError, APIError, ValueError) as exc:
            warnings.warn(
                f"Strict JSON retry failed ({exc}); using current-run factor scoring.",
                RuntimeWarning,
                stacklevel=2,
            )

    result = validate_sdlc_result(result, requirements, project_context)
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
    parser.add_argument(
        "--context", default=None, help="Optional project/team context."
    )
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
