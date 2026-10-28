#!/usr/bin/env python3
"""Run DSA4213 Group 15 M1-M5 experiment.

Requires the local/integration-test branch (all sub-branches merged).
API key must be set in env or .env — never commit .env.

Quick start:
    # Dry run (no API key, no network, mock LLM):
    python -m src.runner.run_experiment --dry-run

    # Real run (50 examples, $0.50 cap):
    python -m src.runner.run_experiment --n 50 --max-cost-usd 0.50

    # Resume interrupted run:
    python -m src.runner.run_experiment --n 50 --run-name run_20261029_120000

    # Retry errored pairs from a previous run:
    python -m src.runner.run_experiment --n 50 --run-name run_20261029_120000 --retry-errors

    # Regenerate outputs without re-running (no API calls):
    python -m src.runner.run_experiment --run-name run_20261029_120000 --report-only

    # Run only M3/M4/M5, reusing drafts from a previous run:
    python -m src.runner.run_experiment --n 50 --methods M3,M4,M5 \
        --reuse-drafts-from run_20261029_120000
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone

_VALID_METHODS = {"M1", "M2", "M3", "M4", "M5"}


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DSA4213 Group 15 M1-M5 experiment runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="openai/gpt-4o-mini",
                   help="OpenRouter model ID (openai/gpt-4o-mini recommended)")
    p.add_argument("--n", type=int, default=50,
                   help="Number of eval examples after dev exclusion")
    p.add_argument("--max-cost-usd", type=float, default=0.50, dest="max_cost_usd",
                   help="Hard budget cap in USD (abort if exceeded)")
    p.add_argument("--query-include-question", action=argparse.BooleanOptionalAction,
                   default=True, dest="query_include_question",
                   help="Prepend question text to M4/M5 retrieval queries")
    p.add_argument("--run-name", default=None, dest="run_name",
                   help="Output subdir under results/runs/ (default: run_<timestamp>)")
    p.add_argument("--retry-errors", action="store_true", dest="retry_errors",
                   help="Retry (example_id, method_id) pairs whose latest attempt is 'error'")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Use mock LLM with 2 synthetic examples; no API key required")
    p.add_argument("--report-only", action="store_true", dest="report_only",
                   help="Regenerate outputs from an existing run dir; no API calls. "
                        "Requires --run-name.")
    p.add_argument("--reuse-drafts-from", default=None, dest="reuse_drafts_from",
                   metavar="RUN_NAME",
                   help="Copy drafts.jsonl from results/runs/<RUN_NAME> into the current run dir "
                        "before running (avoids re-generating drafts).")
    p.add_argument("--methods", default=None,
                   help="Comma-separated method IDs to run, e.g. M3,M4,M5 (default: all).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    run_name = args.run_name or f"run_{_now_tag()}"
    out_dir = pathlib.Path("results") / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = str(out_dir / "records.jsonl")

    # Validate --methods
    methods_list = None
    if args.methods:
        methods_list = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
        invalid = [m for m in methods_list if m not in _VALID_METHODS]
        if invalid:
            print(f"ERROR: --methods contains invalid IDs {invalid}. "
                  f"Valid: {sorted(_VALID_METHODS)}")
            sys.exit(1)

    print(f"[run_experiment] output : {out_dir}")
    print(f"[run_experiment] model  : {args.model}")
    print(f"[run_experiment] dry_run: {args.dry_run}")
    if methods_list:
        print(f"[run_experiment] methods: {','.join(methods_list)}")
    if not args.report_only:
        print(f"[run_experiment] budget : ${args.max_cost_usd:.2f} USD")
    print()

    # --report-only: regenerate outputs from existing run dir, no API calls
    if args.report_only:
        if not args.run_name:
            print("ERROR: --report-only requires --run-name to identify the run directory.")
            sys.exit(1)
        gold_lookup = _load_gold_lookup(out_dir)
        n_dev_excluded = _load_meta_field(out_dir, "n_dev_excluded", 0)
        _write_outputs(out_dir, gold_lookup, args, n_dev_excluded)
        return

    # --reuse-drafts-from: copy drafts cache from another run
    if args.reuse_drafts_from:
        src_dir = pathlib.Path("results") / "runs" / args.reuse_drafts_from
        src_drafts = src_dir / "drafts.jsonl"
        if not src_drafts.exists():
            print(f"ERROR: no drafts.jsonl found in {src_dir}")
            sys.exit(1)
        dst_drafts = out_dir / "drafts.jsonl"
        shutil.copy(src_drafts, dst_drafts)
        print(f"[run_experiment] reusing drafts from: {args.reuse_drafts_from}")
        print()

    if args.dry_run:
        gold_lookup, n_dev_excluded = _run_dry(args, records_path, out_dir, methods_list)
    else:
        gold_lookup, n_dev_excluded = _run_real(args, records_path, out_dir, methods_list)

    _write_outputs(out_dir, gold_lookup, args, n_dev_excluded)


# ---------------------------------------------------------------------------
# gold_lookup persistence (needed for --report-only)
# ---------------------------------------------------------------------------

def _save_gold_lookup(out_dir: pathlib.Path, gold_lookup: dict) -> None:
    path = out_dir / "gold_lookup.json"
    path.write_text(json.dumps(gold_lookup, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_gold_lookup(out_dir: pathlib.Path) -> dict:
    path = out_dir / "gold_lookup.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: gold_lookup.json not found in {out_dir}. "
              "Run the experiment first (without --report-only).")
        sys.exit(1)


def _load_meta_field(out_dir: pathlib.Path, field: str, default):
    path = out_dir / "metadata.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        return meta.get(field, default)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _run_dry(args, records_path, out_dir: pathlib.Path, methods_list):
    from src.runner.mocks import (
        MockDraftPort, MockRetrievePort, MockClaimsPort, MockVerificationPort,
        MockLLMClient, MockQAInput, MockSentence, MockQAGold,
    )
    from src.runner.runner import ExperimentRunner

    examples = [
        MockQAInput(
            example_id="dry_0",
            question="What is the capital of France?",
            context=[MockSentence("France", 0, "Paris is the capital of France.")],
        ),
        MockQAInput(
            example_id="dry_1",
            question="Who wrote Hamlet?",
            context=[MockSentence("Shakespeare", 0, "William Shakespeare wrote Hamlet.")],
        ),
    ]
    gold_lookup = {
        "dry_0": {"gold_answer": "Paris",
                  "gold_supporting_facts": [{"title": "France", "sentence_id": 0}]},
        "dry_1": {"gold_answer": "William Shakespeare",
                  "gold_supporting_facts": [{"title": "Shakespeare", "sentence_id": 0}]},
    }

    runner = ExperimentRunner(
        draft_port=MockDraftPort(),
        retrieve_port=MockRetrievePort(),
        claims_port=MockClaimsPort(),
        verification_port=MockVerificationPort(),
        llm=MockLLMClient(model=args.model),
        top_k=5,
        m4_top_k=5,
        m5_top_k_per_claim=5,
        query_include_question=args.query_include_question,
    )

    progress_fn = _make_progress_fn(len(examples))
    runner.run(
        examples,
        records_path,
        model_settings={"model": args.model, "temperature": 0.0},
        retry_errors=args.retry_errors,
        methods=methods_list,
        progress_fn=progress_fn,
    )
    print("[DRY RUN] No API calls made. Schema verified.")
    _save_gold_lookup(out_dir, gold_lookup)
    return gold_lookup, 0


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------

def _run_real(args, records_path, out_dir: pathlib.Path, methods_list):
    try:
        from src.data.contract_loader import load_hotpotqa_contract
        from src.core.llm_client import OpenRouterLLMClient
        from src.core.budget import BudgetGuard, BudgetConfig
        from src.retrieval.contract_adapter import ContractEvidenceRetriever
        from src.claims.extractor import AtomicClaimExtractor
        import src.verification as _verification_module
        import src.agents.revision as _revision_module
        from src.agents.answer import AnswerAgent, AgentConfig
        from src.runner.adapters import AAnswerAdapter, DVerificationAdapter
        from src.runner.runner import ExperimentRunner
    except ImportError as exc:
        print(f"ERROR: missing dependency — {exc}")
        print("Run from local/integration-test with all sub-branches merged.")
        sys.exit(1)

    from src.evaluation.manifest import load_id_manifest, filter_out_ids, assert_disjoint

    # --- Dev/eval split ---
    dev_ids = load_id_manifest("research/B_dev_50_ids.json")
    load_limit = len(dev_ids) + args.n + 20  # overshoot to guarantee enough after filter
    print(f"Loading up to {load_limit} HotpotQA validation examples...")
    all_examples, gold_lookup_raw = load_hotpotqa_contract("validation", limit=load_limit)
    eval_examples, n_removed = filter_out_ids(all_examples, dev_ids)
    eval_examples = eval_examples[: args.n]
    assert_disjoint(dev_ids, {ex.example_id for ex in eval_examples})
    print(f"Eval set: {len(eval_examples)} examples ({n_removed} dev examples excluded)")

    # Normalise QAGold objects to plain dicts
    gold_lookup = {}
    for eid, g in gold_lookup_raw.items():
        if hasattr(g, "gold_answer"):
            facts = g.gold_supporting_facts or []
            fact_dicts = [
                {"title": f[0], "sentence_id": f[1]} if isinstance(f, tuple)
                else f
                for f in facts
            ]
            gold_lookup[eid] = {"gold_answer": g.gold_answer, "gold_supporting_facts": fact_dicts}
        else:
            gold_lookup[eid] = g

    # --- API key (never log) ---
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Add it to env or a .env file.")
        sys.exit(1)

    # --- LLM with budget guard ---
    budget_guard = BudgetGuard(BudgetConfig(
        max_calls=len(eval_examples) * 5 * 6,
        max_cost_usd=args.max_cost_usd,
        min_key_remaining_usd=0.0,
    ))
    llm = OpenRouterLLMClient(api_key=api_key, budget_guard=budget_guard)

    # --- Ports ---
    agent_config = AgentConfig(model=args.model)
    draft_port = AAnswerAdapter(AnswerAgent(agent_config))

    retrieve_port = ContractEvidenceRetriever()

    claims_port = AtomicClaimExtractor()

    verification_port = DVerificationAdapter(
        verification_module=_verification_module,
        revision_module=_revision_module,
        model=args.model,
        temperature=0.0,
    )

    # --- Runner ---
    runner = ExperimentRunner(
        draft_port=draft_port,
        retrieve_port=retrieve_port,
        claims_port=claims_port,
        verification_port=verification_port,
        llm=llm,
        top_k=5,
        m4_top_k=5,
        m5_top_k_per_claim=5,
        query_include_question=args.query_include_question,
    )

    progress_fn = _make_progress_fn(len(eval_examples))
    runner.run(
        eval_examples,
        records_path,
        model_settings={"model": args.model, "temperature": 0.0},
        retry_errors=args.retry_errors,
        methods=methods_list,
        progress_fn=progress_fn,
    )
    _save_gold_lookup(out_dir, gold_lookup)
    return gold_lookup, n_removed


# ---------------------------------------------------------------------------
# Progress callback factory
# ---------------------------------------------------------------------------

def _make_progress_fn(total: int):
    """Return a progress callback that prints N/total | $cumulative_cost | elapsed."""
    cost_acc = [0.0]
    seen_draft_eids: set = set()

    def _progress(idx, _total, eid, example_recs, elapsed):
        # Accumulate draft_usage once per example (draft is shared across methods)
        if eid not in seen_draft_eids:
            seen_draft_eids.add(eid)
            for rec in example_recs[:1]:  # draft_usage is identical on all recs
                for call in (getattr(rec, "draft_usage", None) or []):
                    cost_acc[0] += call.get("cost_usd") or 0.0
        # Accumulate method usage from all records
        for rec in example_recs:
            for call in (getattr(rec, "usage", None) or []):
                cost_acc[0] += call.get("cost_usd") or 0.0

        if elapsed >= 60:
            elapsed_s = f"{int(elapsed // 60)}m{elapsed % 60:04.1f}s"
        else:
            elapsed_s = f"{elapsed:.1f}s"
        n = idx + 1
        print(f"  [{n}/{total}] {eid} | ${cost_acc[0]:.4f} | {elapsed_s}", flush=True)

    return _progress


# ---------------------------------------------------------------------------
# Post-run outputs (reads ALL records from file — includes retries)
# ---------------------------------------------------------------------------

def _write_outputs(out_dir: pathlib.Path, gold_lookup, args, n_dev_excluded: int = 0) -> None:
    from src.runner.records import load_records, latest_records
    from src.evaluation.answer_metrics import exact_match, token_f1, export_per_example_csv
    from src.evaluation.cost_metrics import aggregate_usage_by_method, aggregate_draft_usage, estimate_budget

    # Load ALL records from file so retry history is included
    all_records = load_records(str(out_dir / "records.jsonl"))
    latest = latest_records(all_records)

    # Group by method
    by_method: dict[str, list] = defaultdict(list)
    for r in latest:
        by_method[r.method_id].append(r)

    # Per-method cost stats across all attempts (for accurate retry waste)
    cost_stats = aggregate_usage_by_method(all_records)

    print("=== Per-method results ===")
    print(f"{'Method':<6} {'N':>4} {'OK':>4}  {'EM':>6}  {'F1':>6}  "
          f"{'Calls':>5}  {'Cost($)':>9}  {'Waste($)':>9}")
    print("-" * 65)

    cost_summary: dict = {}
    for mid in ["M1", "M2", "M3", "M4", "M5"]:
        recs = by_method.get(mid, [])
        n_ok = sum(1 for r in recs if r.status == "ok")

        em_vals, f1_vals = [], []
        for r in recs:
            if r.status != "ok":
                continue
            pred = ((r.final or {}).get("final_answer")
                    or (r.draft or {}).get("short_answer")
                    or "")
            gold = gold_lookup.get(r.example_id, {})
            gold_ans = gold.get("gold_answer", "") if isinstance(gold, dict) else getattr(gold, "gold_answer", "")
            em_vals.append(exact_match(pred, gold_ans))
            f1_vals.append(token_f1(pred, gold_ans))

        em = sum(em_vals) / len(em_vals) if em_vals else float("nan")
        f1 = sum(f1_vals) / len(f1_vals) if f1_vals else float("nan")

        cs = cost_stats.get(mid, {})
        n_calls = cs.get("llm_calls", 0)
        cost = cs.get("cost_usd_sum") or 0.0
        waste = cs.get("retry_waste_cost_usd")
        cost_summary[mid] = cs

        em_s = f"{em:.3f}" if em == em else "  n/a"
        f1_s = f"{f1:.3f}" if f1 == f1 else "  n/a"
        waste_s = f"{waste:.4f}" if waste is not None else "     n/a"
        print(f"{mid:<6} {len(recs):>4} {n_ok:>4}  {em_s:>6}  {f1_s:>6}  "
              f"{n_calls:>5}  {cost:>9.4f}  {waste_s:>9}")

    method_total = sum(
        (cost_summary.get(m, {}).get("cost_usd_sum") or 0.0) for m in ["M1", "M2", "M3", "M4", "M5"]
    )
    print(f"\nTotal method cost (final attempts, excl. draft): ${method_total:.4f} USD")

    # Draft cost and per-example breakdown
    draft_stats = aggregate_draft_usage(latest)
    n_ex = draft_stats.get("n_examples", 0)
    draft_total = draft_stats.get("cost_usd_sum") or 0.0
    draft_per_ex = draft_total / n_ex if n_ex > 0 else 0.0

    print("\n=== Per-example cost breakdown ===")
    print(f"  Draft (shared)   : ${draft_per_ex:.5f}/example")
    method_per_ex_sum = 0.0
    for mid in ["M1", "M2", "M3", "M4", "M5"]:
        per_ex = (cost_summary.get(mid) or {}).get("per_example_cost_usd")
        if per_ex is not None:
            method_per_ex_sum += per_ex
            print(f"  {mid}              : ${per_ex:.5f}/example")
        else:
            print(f"  {mid}              :       n/a")
    pipeline_per_ex = draft_per_ex + method_per_ex_sum
    print(f"  Full pipeline    : ${pipeline_per_ex:.5f}/example  (draft + M1–M5)")

    # Budget projections
    if latest:
        print("\n=== Budget estimates (draft + methods) ===")
        est50  = estimate_budget(all_records, target_n_examples=50)
        est100 = estimate_budget(all_records, target_n_examples=100)
        est200 = estimate_budget(all_records, target_n_examples=200)
        for label, e in [("N=50 ", est50), ("N=100", est100), ("N=200", est200)]:
            total_est = e.get("total_estimated_cost_usd")
            draft_est = (e.get("draft") or {}).get("estimated_cost_usd")
            total_s  = f"${total_est:.3f}" if total_est is not None else "  n/a"
            draft_s  = f"${draft_est:.3f}" if draft_est is not None else "  n/a"
            print(f"  {label}: {total_s} total  (incl. draft {draft_s})")
        print()
        for mid in ["M1", "M2", "M3", "M4", "M5"]:
            parts = []
            for label, e in [("N=50", est50), ("N=100", est100), ("N=200", est200)]:
                v = (e.get("methods") or {}).get(mid, {}).get("estimated_cost_usd")
                parts.append(f"{label}=${v:.4f}" if v is not None else f"{label}=n/a")
            print(f"  {mid}: {' | '.join(parts)}")

    # Per-example CSV + metadata.json
    csv_path = str(out_dir / "per_example.csv")
    run_meta: dict = {
        "model": args.model,
        "query_include_question": args.query_include_question,
        "m4_top_k": 5,
        "m5_top_k_per_claim": 5,
        "n_dev_excluded": n_dev_excluded,
        "run_timestamp": _now_tag(),
    }
    if getattr(args, "reuse_drafts_from", None):
        run_meta["reused_drafts_from"] = args.reuse_drafts_from
    if getattr(args, "methods", None):
        run_meta["methods"] = args.methods
    export_per_example_csv(
        latest, gold_lookup, csv_path,
        preliminary=True, split="validation",
        extra_meta=run_meta,
    )

    # cost_summary.json
    cost_path = out_dir / "cost_summary.json"
    cost_path.write_text(
        json.dumps(cost_summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nOutputs written to {out_dir}/")
    print(f"  records.jsonl     — {len(all_records)} records ({len(latest)} unique pairs)")
    print(f"  raw_calls.jsonl   — raw LLM call log")
    print(f"  per_example.csv   — per-example EM/F1")
    print(f"  metadata.json     — run metadata")
    print(f"  cost_summary.json — per-method cost breakdown")


if __name__ == "__main__":
    main()
