import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

from contracts.schemas import AnswerDraft
from src.claims.extractor import (
    AtomicClaimExtractor,
    ClaimExtractionError,
)
from src.core.llm_client import (
    LLMClientError,
    OpenRouterLLMClient,
)
from src.core.budget import (
    BudgetConfig,
    BudgetGuard,
    BudgetExceededError,
)


FIELDS = [
    "draft_id",
    "status",
    "claim_count",
    "error_type",
]


class MockLLMClient:
    """Simulate extraction without real API calls."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = 0

    def generate(
        self, *, stage, messages, model,
        temperature, max_tokens
    ):
        self.calls += 1

        if stage != "claim_extraction":
            raise RuntimeError("Mock repair not supported")

        content = messages[-1]["content"]

        for explanation, claims in self.responses.items():
            if not explanation:
                continue

            if content.endswith(explanation):
                return SimpleNamespace(
                    text=json.dumps({"claims": claims})
                )

        raise RuntimeError("No mock response configured")


def percentile_nearest_rank(values, percent):
    ordered = sorted(values)
    if not ordered:
        return None

    index = max(
        0,
        math.ceil(percent / 100 * len(ordered)) - 1,
    )
    return ordered[index]


def summarize(records):
    counts = [
        row["claim_count"]
        for row in records
        if row["status"] == "success"
    ]

    if not counts:
        return {
            "total": len(records),
            "success": 0,
            "failed": len(records),
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }

    return {
        "total": len(records),
        "success": len(counts),
        "failed": len(records) - len(counts),
        "mean": statistics.mean(counts),
        "median": statistics.median(counts),
        "p90": percentile_nearest_rank(counts, 90),
        "max": max(counts),
    }


def analyze(drafts, extractor, llm, max_examples):
    """Keep compatibility with existing unit tests."""
    records = []

    for draft in drafts[:max_examples]:
        try:
            claims = extractor.extract_claims(draft, llm)
            records.append({
                "draft_id": draft.draft_id,
                "status": "success",
                "claim_count": len(claims),
                "error_type": "",
            })

        except (
            LLMClientError,
            ClaimExtractionError,
        ) as exc:
            records.append({
                "draft_id": draft.draft_id,
                "status": "error",
                "claim_count": None,
                "error_type": type(exc).__name__,
            })

    return records


def save_csv(records, output_path):
    """Write CSV atomically to reduce corruption risk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(records)

    temporary.replace(path)


def load_drafts(path):
    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")

    drafts = [
        AnswerDraft(
            draft_id=item["draft_id"],
            short_answer=item["short_answer"],
            explanation=item["explanation"],
        )
        for item in data
    ]

    ids = [draft.draft_id for draft in drafts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate draft_id detected")

    return drafts


def make_mock_responses(drafts):
    return {
        draft.explanation: [
            sentence.strip()
            for sentence in draft.explanation.split(".")
            if sentence.strip()
        ]
        for draft in drafts
    }


def make_fingerprint(drafts, mode, extractor):
    """Identify input data and extraction configuration."""
    prompt_path = getattr(extractor, "prompt_path", None)

    if prompt_path is None:
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "prompts"
            / "claim_extraction"
            / "system.txt"
        )

    prompt_path = Path(prompt_path)
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Claim extraction prompt not found: {prompt_path}"
        )

    payload = {
        "mode": mode,
        "drafts": [
            {
                "draft_id": d.draft_id,
                "short_answer": d.short_answer,
                "explanation": d.explanation,
            }
            for d in drafts
        ],
        "model": extractor.model,
        "temperature": extractor.temperature,
        "max_tokens": extractor.max_tokens,
        "max_repair_attempts": extractor.max_repair_attempts,
        "prompt_sha256": hashlib.sha256(
            prompt_path.read_bytes()
        ).hexdigest(),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def metadata_path(output):
    return Path(str(output) + ".meta.json")


def save_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_existing_records(output):
    with Path(output).open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDS:
            raise ValueError(
                "Existing CSV has incompatible columns"
            )

        records = []
        seen = set()

        for row in reader:
            draft_id = row["draft_id"]

            if draft_id in seen:
                raise ValueError(
                    "Duplicate draft_id in existing CSV"
                )
            seen.add(draft_id)

            if row["status"] == "success":
                count = int(row["claim_count"])
                if count < 0:
                    raise ValueError("Negative claim count")
                row["claim_count"] = count

            elif row["status"] == "error":
                row["claim_count"] = None

            else:
                raise ValueError(
                    "Invalid status in existing CSV"
                )

            records.append(row)

    return records


def prepare_run(output, fingerprint, resume):
    output = Path(output)
    meta_file = metadata_path(output)

    if resume:
        if not output.exists() or not meta_file.exists():
            raise ValueError(
                "Resume requires CSV and metadata files"
            )

        metadata = json.loads(
            meta_file.read_text(encoding="utf-8")
        )

        if metadata.get("fingerprint") != fingerprint:
            raise ValueError(
                "Input or extraction configuration changed; "
                "use a new output path"
            )

        return load_existing_records(output)

    if output.exists() or meta_file.exists():
        raise FileExistsError(
            "Output already exists. Use --resume with "
            "matching metadata, or choose a new output path."
        )

    save_metadata(meta_file, {
        "fingerprint": fingerprint,
        "format_version": 1,
    })
    return []


def run_batch(
    drafts,
    extractor,
    llm,
    output,
    max_examples,
    existing=None,
    budget=None,
):
    """Process drafts and checkpoint after each result."""
    records = list(existing or [])
    completed = {
        row["draft_id"]
        for row in records
        if row["status"] == "success"
    }

    # Failed rows are not automatically retried.
    recorded = {row["draft_id"] for row in records}

    for draft in drafts[:max_examples]:
        if draft.draft_id in completed:
            continue

        if draft.draft_id in recorded:
            continue

        try:
            claims = extractor.extract_claims(draft, llm)

            row = {
                "draft_id": draft.draft_id,
                "status": "success",
                "claim_count": len(claims),
                "error_type": "",
            }

        except BudgetExceededError:
            print(
                "Budget limit reached. "
                "Stopping without further API calls."
            )
            break

        except (
            LLMClientError,
            ClaimExtractionError,
        ) as exc:
            row = {
                "draft_id": draft.draft_id,
                "status": "error",
                "claim_count": None,
                "error_type": type(exc).__name__,
            }

        records.append(row)
        recorded.add(draft.draft_id)
        save_csv(records, output)

        print(
            f"{row['draft_id']}: "
            f"{row['status']} | "
            f"claims={row['claim_count']}"
        )

    if not Path(output).exists():
        save_csv(records, output)

    return records


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-examples", type=int, default=5
    )
    parser.add_argument(
        "--max-calls", type=int, default=6
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=0.02
    )
    parser.add_argument(
        "--min-key-remaining-usd",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--model", default="openai/gpt-4o-mini"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512
    )
    parser.add_argument(
        "--resume", action="store_true"
    )
    parser.add_argument(
        "--confirm-real", action="store_true"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.max_examples < 1:
        parser.error("--max-examples must be positive")

    if args.mode == "real":
        if not args.confirm_real:
            parser.error(
                "Real API mode requires --confirm-real"
            )

        if args.max_calls < 1:
            parser.error("--max-calls must be positive")

        if (
            not math.isfinite(args.max_cost_usd)
            or args.max_cost_usd <= 0
        ):
            parser.error(
                "--max-cost-usd must be positive and finite"
            )

        if (
            not math.isfinite(args.min_key_remaining_usd)
            or args.min_key_remaining_usd < 0
        ):
            parser.error(
                "--min-key-remaining-usd must be "
                "non-negative and finite"
            )

    drafts = load_drafts(args.input)

    extractor = AtomicClaimExtractor(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_repair_attempts=(
            0 if args.mode == "mock" else 1
        ),
    )

    fingerprint = make_fingerprint(
        drafts, args.mode, extractor
    )

    existing = prepare_run(
        args.output, fingerprint, args.resume
    )

    budget = None

    if args.mode == "mock":
        llm = MockLLMClient(
            make_mock_responses(drafts)
        )

    else:
        budget = BudgetGuard(
            BudgetConfig(
                max_calls=args.max_calls,
                max_cost_usd=args.max_cost_usd,
                min_key_remaining_usd=(
                    args.min_key_remaining_usd
                ),
            )
        )

        llm = OpenRouterLLMClient(
            budget_guard=budget
        )

    try:
        records = run_batch(
            drafts=drafts,
            extractor=extractor,
            llm=llm,
            output=args.output,
            max_examples=args.max_examples,
            existing=existing,
            budget=budget,
        )

    finally:
        if budget is not None:
            print("\n=== BUDGET SUMMARY ===")
            print(f"API attempts: {budget.calls}")
            print(
                "Budget accounted USD: "
                f"{budget.spent_usd:.6f}"
            )
            print(
                "Reserved USD: "
                f"{budget.reserved_usd:.6f}"
            )

    summary = summarize(records)

    print("\n=== CLAIM DISTRIBUTION ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"Results saved to: {args.output}")
    print(f"MODE: {args.mode.upper()}")

    if args.mode == "mock":
        print(f"Mock LLM calls: {llm.calls}")
        print("NO REAL API COST")


if __name__ == "__main__":
    main()