import json
from pathlib import Path

from contracts.interfaces import LLMClient
from contracts.schemas import AnswerDraft, AtomicClaim
from src.core.budget import BudgetExceededError
from src.core.llm_client import LLMClientError


DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "claim_extraction"
    / "system.txt"
)


class ClaimExtractionError(Exception):
    """Raised when claim output cannot be parsed or validated."""
    pass


class AtomicClaimExtractor:
    """Extract independent factual claims using an LLM."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
        prompt_path: str | Path | None = None,
        max_repair_attempts: int = 1,
    ):
        if not model or not model.strip():
            raise ValueError("Model ID cannot be empty")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if max_repair_attempts not in (0, 1):
            raise ValueError(
                "max_repair_attempts must be 0 or 1"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_repair_attempts = max_repair_attempts

        path = (
            Path(prompt_path)
            if prompt_path is not None
            else DEFAULT_PROMPT_PATH
        )

        self.system_prompt = path.read_text(
            encoding="utf-8"
        ).strip()

        if not self.system_prompt:
            raise ValueError(
                "Claim extraction prompt is empty"
            )

    @staticmethod
    def _parse_claims(text: str) -> list[str]:
        """Parse JSON and validate the claims schema."""

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ClaimExtractionError(
                "Invalid JSON returned by claim extractor"
            ) from exc

        if not isinstance(data, dict):
            raise ClaimExtractionError(
                "Expected a JSON object"
            )

        if set(data.keys()) != {"claims"}:
            raise ClaimExtractionError(
                "Expected exactly one key: claims"
            )

        raw_claims = data["claims"]

        if not isinstance(raw_claims, list):
            raise ClaimExtractionError(
                "claims must be a list"
            )

        claims = []
        seen = set()

        for item in raw_claims:
            if not isinstance(item, str):
                raise ClaimExtractionError(
                    "Each claim must be a string"
                )

            claim = item.strip()

            if not claim:
                raise ClaimExtractionError(
                    "Claim text cannot be empty"
                )

            # Deduplicate exact textual equivalents.
            normalized = " ".join(
                claim.casefold().split()
            )

            if normalized in seen:
                continue

            seen.add(normalized)
            claims.append(claim)

        return claims

    def _call_llm(
        self,
        llm: LLMClient,
        messages: list[dict[str, str]],
        stage: str,
    ) -> str:
        """Call the shared LLM client and preserve API errors."""

        try:
            response = llm.generate(
                stage=stage,
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        except LLMClientError:
            # Preserve the original LLMClient error type.
            raise

        except BudgetExceededError:
            raise

        except Exception as exc:
            raise ClaimExtractionError(
                "Unexpected LLM extraction failure"
            ) from exc

        if not isinstance(response.text, str):
            raise ClaimExtractionError(
                "LLM response text must be a string"
            )

        return response.text

    def _repair_json(
        self,
        llm: LLMClient,
        original_output: str,
        explanation: str,
        short_answer: str,
    ) -> list[str]:
        """Attempt one bounded LLM-based JSON repair."""

        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON repair assistant. "
                    "Repair the supplied claim extraction output "
                    "into valid JSON with exactly one key: claims. "
                    "The claims value must be a list of strings. "
                    "Preserve the factual meaning of the original "
                    "explanation. Do not add, verify, or correct facts. "
                    "Do not invent claims when the output is ambiguous. "
                    "Return only the JSON object, without Markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Original explanation:\n"
                    f"{explanation}\n\n"
                    "Short answer (context only):\n"
                    f"{short_answer}\n\n"
                    "Invalid extraction output:\n"
                    f"{original_output}\n\n"
                    "Repair the JSON structure and claim values. "
                    "If the intended claims cannot be recovered "
                    "reliably, return an explicit error message "
                    "instead of inventing or deleting facts."
                ),
            },
        ]

        # Repair is a separate, budgeted LLM request.
        repaired_text = self._call_llm(
            llm=llm,
            messages=repair_messages,
            stage="claim_extraction_repair",
        )

        # A repair that remains invalid must fail explicitly.
        return self._parse_claims(repaired_text)

    def extract_claims(
        self,
        draft: AnswerDraft,
        llm: LLMClient,
    ) -> list[AtomicClaim]:
        """Extract atomic claims from an AnswerDraft."""

        explanation = draft.explanation.strip()

        # Empty input is a valid zero-claim case.
        if not explanation:
            return []

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Short answer (context only):\n"
                    f"{draft.short_answer}\n\n"
                    "Explanation to analyze:\n"
                    f"{explanation}"
                ),
            },
        ]

        # Step 1: Generate the initial extraction.
        text = self._call_llm(
            llm=llm,
            messages=messages,
            stage="claim_extraction",
        )

        # Step 2: Validate or perform one repair.
        try:
            claim_texts = self._parse_claims(text)

        except ClaimExtractionError as original_error:
            if self.max_repair_attempts == 0:
                raise

            try:
                claim_texts = self._repair_json(
                    llm=llm,
                    original_output=text,
                    explanation=explanation,
                    short_answer=draft.short_answer,
                )

            except (LLMClientError, BudgetExceededError):
                # Repair API failures must propagate unchanged.
                raise

            except ClaimExtractionError as repair_error:
                raise ClaimExtractionError(
                    "Claim extraction JSON repair failed"
                ) from repair_error

            # Do not silently interpret malformed output as
            # a genuine empty extraction.
            if not claim_texts:
                raise ClaimExtractionError(
                    "Repair returned an ambiguous empty result"
                ) from original_error

        # Step 3: Convert to public AtomicClaim objects.
        return [
            AtomicClaim(
                claim_id=f"c{i}",
                claim_text=claim_text,
                source_span=None,
            )
            for i, claim_text in enumerate(
                claim_texts,
                start=1,
            )
        ]
