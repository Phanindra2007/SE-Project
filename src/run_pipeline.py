"""Run requirements generation and SDLC selection as one local-LLM pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

from src.step1_generate_reqs import generate_requirements
from src.step2_select_sdlc import select_sdlc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path from the project folder unless it is already absolute."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def output_name(prefix: str | None, default_name: str) -> str:
    """Add a safe prefix to an output filename when provided."""
    if not prefix:
        return default_name
    safe_prefix = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in prefix.strip()
    ).strip("-")
    if not safe_prefix:
        raise ValueError("prefix must contain at least one letter or number.")
    return f"{safe_prefix}_{default_name}"


def run_pipeline(
    functionality: str,
    *,
    data_dir: str | Path = PROJECT_ROOT / "data" / "banking",
    output_dir: str | Path = PROJECT_ROOT / "outputs",
    context: str | None = None,
    prefix: str | None = None,
) -> tuple[dict[str, Any], dict]:
    """Generate requirements and then choose an SDLC model."""
    functionality = functionality.strip()
    if not functionality:
        raise ValueError("functionality must not be empty.")

    data_path = resolve_project_path(data_dir)
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    requirements_path = output_path / output_name(prefix, "step1_requirements.json")
    sdlc_json_path = output_path / output_name(prefix, "step2_sdlc_justification.json")
    sdlc_markdown_path = output_path / output_name(
        prefix, "step2_sdlc_justification.md"
    )

    print("Step 1: Generating grounded requirements...")
    requirements = generate_requirements(
        functionality,
        data_dir=data_path,
        output_path=requirements_path,
    )
    print(f"  Generated {len(requirements['requirements'])} requirements")

    print("Step 2: Selecting an SDLC model...")
    recommendation = select_sdlc(
        requirements_path,
        context,
        json_output_path=sdlc_json_path,
        markdown_output_path=sdlc_markdown_path,
    )
    print(f"  Recommended SDLC: {recommendation['recommended_sdlc']}")

    print("Outputs:")
    print(f"  Requirements JSON: {requirements_path}")
    print(f"  SDLC JSON: {sdlc_json_path}")
    print(f"  SDLC Markdown: {sdlc_markdown_path}")

    return requirements, recommendation


def build_parser() -> argparse.ArgumentParser:
    """Set up the command-line options for the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the banking requirements-to-SDLC local LLM pipeline."
    )
    parser.add_argument(
        "--functionality",
        required=True,
        help='Target functionality, for example "Domestic Instant Fund Transfer".',
    )
    parser.add_argument(
        "--data-dir",
        default="data/banking",
        help="Directory containing banking Markdown sources.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for Step 1 and Step 2 output files.",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Optional project/team/schedule notes for Step 2.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional filename prefix, for example 'test1'.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the full pipeline from the command line."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        run_pipeline(
            args.functionality,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            context=args.context,
            prefix=args.prefix,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
