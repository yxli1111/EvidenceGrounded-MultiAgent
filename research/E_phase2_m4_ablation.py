"""Phase 2 M4 evidence-volume ablation analysis.

Usage
-----
    python research/E_phase2_m4_ablation.py \\
        --runs top5=run_name_top5 top10=run_name_top10 full=run_name_full \\
        [--out-dir results/analysis/E_phase2_m4]

Each run must contain at least M1 and M4 records.  Examples are matched
by example_id; only IDs present in ALL specified runs are included.

Outputs (written to --out-dir):
    summary.csv / summary.json  — per-mode EM, F1, ΔEM vs M1, CI, label
                                   distribution, cost, latency, SF recall,
                                   failure counts.
    pairwise.csv / pairwise.json — pairwise ΔEM + CI for every mode pair.
    em_by_mode.png              — EM with 95 % CI bar chart.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Shared answer normalisation (mirrors answer_metrics.py)
# ---------------------------------------------------------------------------

import re
import string


def _normalize(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _em(pred: str | None, gold: str | None) -> float:
    return float(_normalize(pred) == _normalize(gold))


def _token_f1(pred: str | None, gold: str | None) -> float:
    from collections import Counter
    pn, gn = _normalize(pred), _normalize(gold)
    _yes_no = {"yes", "no", "noanswer"}
    if pn in _yes_no or gn in _yes_no:
        return 1.0 if pn == gn else 0.0
    pt, gt = pn.split() if pn else [], gn.split() if gn else []
    if not gt and not pt:
        return 1.0
    if not gt or not pt:
        return 0.0
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pt), common / len(gt)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return (lower, upper) percentile bootstrap CI for the mean."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    boot_means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_i = int(alpha / 2 * n_boot)
    hi_i = int((1 - alpha / 2) * n_boot) - 1
    return boot_means[lo_i], boot_means[hi_i]


def _bootstrap_delta_ci(
    vals_a: list[float],
    vals_b: list[float],
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired bootstrap: return (delta_mean, ci_lo, ci_hi) for b - a."""
    assert len(vals_a) == len(vals_b), "Paired bootstrap requires equal-length lists"
    rng = random.Random(seed)
    n = len(vals_a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    delta_obs = sum(vals_b) / n - sum(vals_a) / n
    boot_deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(vals_a[i] for i in idx) / n
        db = sum(vals_b[i] for i in idx) / n
        boot_deltas.append(db - da)
    boot_deltas.sort()
    lo_i = int(alpha / 2 * n_boot)
    hi_i = int((1 - alpha / 2) * n_boot) - 1
    return delta_obs, boot_deltas[lo_i], boot_deltas[hi_i]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_run(run_dir: pathlib.Path) -> tuple[dict, dict]:
    """Return (records_by_eid_mid, gold_lookup)."""
    records_path = run_dir / "records.jsonl"
    gold_path = run_dir / "gold_lookup.json"

    if not records_path.exists():
        sys.exit(f"ERROR: {records_path} not found")
    if not gold_path.exists():
        sys.exit(f"ERROR: {gold_path} not found")

    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    # Dedup: keep highest attempt per (eid, mid)
    best: dict[tuple, dict] = {}
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["example_id"], rec["method_id"])
            existing = best.get(key)
            if existing is None or rec.get("attempt", 1) > existing.get("attempt", 1):
                best[key] = rec

    return best, gold


def _get_pred(rec: dict) -> str:
    final = rec.get("final") or {}
    draft = rec.get("draft") or {}
    if isinstance(final, dict) and final.get("final_answer"):
        return final["final_answer"]
    if isinstance(draft, dict) and draft.get("short_answer"):
        return draft["short_answer"]
    return ""


# ---------------------------------------------------------------------------
# Per-record helpers
# ---------------------------------------------------------------------------

def _sf_recall(rec: dict, gold_entry: dict | None) -> float | None:
    if not gold_entry:
        return None
    gold_sfs = gold_entry.get("gold_supporting_facts") or []
    if not gold_sfs:
        return None
    evidence = rec.get("evidence") or {}
    if not isinstance(evidence, dict):
        return None
    retrieved: set[tuple] = set()
    for ev_list in evidence.values():
        for ev in (ev_list or []):
            if isinstance(ev, dict):
                retrieved.add((ev.get("doc_title", ""), ev.get("sentence_id", -1)))
    if not retrieved:
        return None
    n_hit = sum(
        1 for sf in gold_sfs
        if isinstance(sf, dict) and (sf.get("title", ""), sf.get("sentence_id", -1)) in retrieved
    )
    return n_hit / len(gold_sfs)


def _feedback_counts(rec: dict) -> tuple[int, int, int]:
    fb = rec.get("feedback") or {}
    items = fb.get("items") or [] if isinstance(fb, dict) else []
    n_sup = sum(1 for i in items if isinstance(i, dict) and i.get("label") == "SUPPORTED")
    n_uns = sum(1 for i in items if isinstance(i, dict) and i.get("label") == "UNSUPPORTED")
    n_con = sum(1 for i in items if isinstance(i, dict) and i.get("label") == "CONTRADICTED")
    return n_sup, n_uns, n_con


def _n_evidence(rec: dict) -> int:
    ev = rec.get("evidence") or {}
    if isinstance(ev, dict):
        return sum(len(v) for v in ev.values() if isinstance(v, list))
    return 0


def _method_cost(rec: dict) -> float:
    return sum((u.get("cost_usd") or 0.0) for u in (rec.get("usage") or []) if isinstance(u, dict))


def _draft_cost(rec: dict) -> float:
    return sum((u.get("cost_usd") or 0.0) for u in (rec.get("draft_usage") or []) if isinstance(u, dict))


def _latency(rec: dict) -> float:
    return sum((u.get("latency_seconds") or 0.0) for u in (rec.get("usage") or []) if isinstance(u, dict))


def _answer_changed(m1_rec: dict, m4_rec: dict) -> bool | None:
    if m4_rec.get("status") != "ok":
        return None
    m4_final = (m4_rec.get("final") or {}).get("final_answer", "")
    m1_draft = (m1_rec.get("draft") or {}).get("short_answer", "")
    return m4_final != m1_draft


# ---------------------------------------------------------------------------
# Mode analysis
# ---------------------------------------------------------------------------

def _analyse_mode(
    mode_label: str,
    records: dict[tuple, dict],
    gold: dict,
    m1_em_by_eid: dict[str, float],
    common_eids: list[str],
) -> dict:
    """Compute all metrics for one mode across common_eids."""
    em_vals, f1_vals, delta_em_vals = [], [], []
    n_ok = n_error = 0
    n_improved = n_worsened = 0
    n_changed = 0
    sups, unss, cons = [], [], []
    n_ev_vals, sfr_vals = [], []
    cost_vals, latency_vals = [], []
    draft_cost_vals = []
    errors: dict[str, int] = defaultdict(int)

    for eid in common_eids:
        rec = records.get((eid, "M4"))
        m1_rec = records.get((eid, "M1"))
        if rec is None:
            continue
        gold_entry = gold.get(eid) or {}
        gold_ans = (gold_entry.get("gold_answer") if isinstance(gold_entry, dict)
                    else getattr(gold_entry, "gold_answer", "")) or ""

        if rec.get("status") == "ok":
            n_ok += 1
            pred = _get_pred(rec)
            ev = _em(pred, gold_ans)
            fv = _token_f1(pred, gold_ans)
            em_vals.append(ev)
            f1_vals.append(fv)

            m1_em = m1_em_by_eid.get(eid, 0.0)
            d = ev - m1_em
            delta_em_vals.append(d)
            if d > 0:
                n_improved += 1
            elif d < 0:
                n_worsened += 1

            if m1_rec and _answer_changed(m1_rec, rec):
                n_changed += 1

            s, u, c = _feedback_counts(rec)
            sups.append(s); unss.append(u); cons.append(c)

            n_ev = _n_evidence(rec)
            n_ev_vals.append(n_ev)

            sfr = _sf_recall(rec, gold_entry)
            if sfr is not None:
                sfr_vals.append(sfr)

            cost_vals.append(_method_cost(rec))
            draft_cost_vals.append(_draft_cost(rec))
            latency_vals.append(_latency(rec))
        else:
            n_error += 1
            em_vals.append(0.0)
            f1_vals.append(0.0)
            delta_em_vals.append(0.0 - m1_em_by_eid.get(eid, 0.0))
            err_msg = (rec.get("error") or "unknown")[:80]
            errors[err_msg] += 1

    n = len(common_eids)
    em_mean = sum(em_vals) / n if n else float("nan")
    f1_mean = sum(f1_vals) / n if n else float("nan")
    delta_mean = sum(delta_em_vals) / n if n else float("nan")
    em_ci = _bootstrap_mean_ci(em_vals) if em_vals else (float("nan"), float("nan"))
    delta_ci = _bootstrap_mean_ci(delta_em_vals) if delta_em_vals else (float("nan"), float("nan"))

    return {
        "mode": mode_label,
        "n": n,
        "n_ok": n_ok,
        "n_error": n_error,
        "em": em_mean,
        "em_ci_lo": em_ci[0],
        "em_ci_hi": em_ci[1],
        "f1": f1_mean,
        "delta_em_vs_m1": delta_mean,
        "delta_em_ci_lo": delta_ci[0],
        "delta_em_ci_hi": delta_ci[1],
        "n_improved": n_improved,
        "n_worsened": n_worsened,
        "n_changed": n_changed,
        "avg_n_supported": sum(sups) / n_ok if n_ok else float("nan"),
        "avg_n_unsupported": sum(unss) / n_ok if n_ok else float("nan"),
        "avg_n_contradicted": sum(cons) / n_ok if n_ok else float("nan"),
        "avg_evidence": sum(n_ev_vals) / n_ok if n_ok else float("nan"),
        # SF recall = 1.0 for full mode by definition (all sentences retrieved)
        "avg_sf_recall": (
            "1.0 (full context — all gold SFs retrieved by definition)"
            if mode_label == "full"
            else (sum(sfr_vals) / len(sfr_vals) if sfr_vals else float("nan"))
        ),
        "avg_method_cost_usd": sum(cost_vals) / n_ok if n_ok else float("nan"),
        "avg_draft_cost_usd": sum(draft_cost_vals) / n_ok if n_ok else float("nan"),
        "avg_latency_s": sum(latency_vals) / n_ok if n_ok else float("nan"),
        "error_breakdown": dict(errors),
        # keep raw EM values for pairwise bootstrap
        "_em_vals": em_vals,
    }


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

def _pairwise(
    label_a: str,
    label_b: str,
    em_a: list[float],
    em_b: list[float],
) -> dict:
    delta, lo, hi = _bootstrap_delta_ci(em_a, em_b)
    return {
        "comparison": f"{label_b} − {label_a}",
        "delta_em": delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": len(em_a),
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(path: pathlib.Path, rows: list[dict], keys: list[str]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(summary_rows: list[dict], out_path: pathlib.Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available; skipping plot.")
        return

    modes = [r["mode"] for r in summary_rows]
    ems   = [r["em"] for r in summary_rows]
    lo    = [r["em"] - r["em_ci_lo"] for r in summary_rows]
    hi    = [r["em_ci_hi"] - r["em"] for r in summary_rows]

    # x-axis labels show evidence quantity
    x_labels = []
    for r in summary_rows:
        if r["mode"] == "top5":
            x_labels.append("5\n(top-5)")
        elif r["mode"] == "top10":
            x_labels.append("10\n(top-10)")
        elif r["mode"] == "full":
            avg_ev = r.get("avg_evidence")
            lbl = f"all\n(full ctx)"
            if isinstance(avg_ev, float) and not _isnan(avg_ev):
                lbl = f"~{avg_ev:.0f}\n(full ctx)"
            x_labels.append(lbl)
        else:
            x_labels.append(r["mode"])

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(range(len(modes)))
    ax.bar(xs, ems, yerr=[lo, hi], capsize=6, color=["#4c72b0", "#dd8452", "#55a868"],
           alpha=0.85, error_kw={"elinewidth": 1.5})
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Evidence volume")
    ax.set_ylabel("Exact Match (EM)")
    ax.set_title("M4 EM by evidence volume (95 % bootstrap CI)")
    ax.set_ylim(0, min(1.0, max(ems) + 0.15))
    for i, (x, y) in enumerate(zip(xs, ems)):
        ax.text(x, y + lo[i] + 0.01, f"{y:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {out_path}")


def _isnan(v) -> bool:
    import math
    try:
        return math.isnan(v)
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 M4 ablation analysis.")
    p.add_argument(
        "--runs", nargs="+", metavar="MODE=RUN_NAME",
        help="e.g. top5=run_name_top5 top10=run_name_top10 full=run_name_full",
    )
    p.add_argument(
        "--runs-dir", default="results/runs",
        help="Root directory containing run subdirs.",
    )
    p.add_argument(
        "--out-dir", default="results/analysis/E_phase2_m4",
        help="Output directory.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.runs:
        print("ERROR: --runs required. e.g. --runs top5=my_run_top5 top10=my_run_top10")
        sys.exit(1)

    # Parse mode=run_name pairs
    mode_runs: list[tuple[str, str]] = []
    for tok in args.runs:
        if "=" not in tok:
            sys.exit(f"ERROR: expected MODE=RUN_NAME, got {tok!r}")
        mode, run_name = tok.split("=", 1)
        mode_runs.append((mode, run_name))

    runs_root = pathlib.Path(args.runs_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    all_records: dict[str, dict[tuple, dict]] = {}
    all_gold: dict[str, dict] = {}
    for mode, run_name in mode_runs:
        run_dir = runs_root / run_name
        recs, gold = _load_run(run_dir)
        all_records[mode] = recs
        all_gold[mode] = gold
        n_m4 = sum(1 for (_, mid) in recs if mid == "M4")
        print(f"  Loaded {mode:6s} ({run_name}): {n_m4} M4 records")

    # Common example IDs present in all runs with both M1 and M4
    eid_sets = []
    for mode, _ in mode_runs:
        recs = all_records[mode]
        eids = {eid for (eid, mid) in recs if mid == "M4"} & {eid for (eid, mid) in recs if mid == "M1"}
        eid_sets.append(eids)
    common_eids = sorted(set.intersection(*eid_sets))
    print(f"  Common examples across all modes: {len(common_eids)}")
    if not common_eids:
        sys.exit("ERROR: no common examples found — check that runs share the same examples.")

    # M1 EM baseline (use first run's M1; all runs share the same draft)
    first_mode = mode_runs[0][0]
    gold_first = all_gold[first_mode]
    m1_em_by_eid: dict[str, float] = {}
    for eid in common_eids:
        m1_rec = all_records[first_mode].get((eid, "M1"))
        if m1_rec and m1_rec.get("status") == "ok":
            gold_entry = gold_first.get(eid) or {}
            gold_ans = (gold_entry.get("gold_answer") or "") if isinstance(gold_entry, dict) else ""
            m1_em_by_eid[eid] = _em(_get_pred(m1_rec), gold_ans)
        else:
            m1_em_by_eid[eid] = 0.0

    m1_em_mean = sum(m1_em_by_eid.values()) / len(m1_em_by_eid) if m1_em_by_eid else float("nan")
    print(f"  M1 EM baseline: {m1_em_mean:.3f}")

    # Per-mode summary
    summary_rows = []
    for mode, _ in mode_runs:
        row = _analyse_mode(
            mode_label=mode,
            records=all_records[mode],
            gold=all_gold[mode],
            m1_em_by_eid=m1_em_by_eid,
            common_eids=common_eids,
        )
        summary_rows.append(row)
        print(f"  {mode:6s} EM={row['em']:.3f} [{row['em_ci_lo']:.3f},{row['em_ci_hi']:.3f}]"
              f"  ΔEM={row['delta_em_vs_m1']:+.3f}  F1={row['f1']:.3f}"
              f"  ok={row['n_ok']}/{row['n']}  ev={row['avg_evidence']:.1f}")

    # Pairwise comparisons
    pairwise_rows = []
    mode_labels = [r["mode"] for r in summary_rows]
    em_by_mode = {r["mode"]: r["_em_vals"] for r in summary_rows}
    for i in range(len(mode_labels)):
        for j in range(i + 1, len(mode_labels)):
            la, lb = mode_labels[i], mode_labels[j]
            pw = _pairwise(la, lb, em_by_mode[la], em_by_mode[lb])
            pairwise_rows.append(pw)
            print(f"  {lb}−{la}: ΔEM={pw['delta_em']:+.3f} [{pw['ci_lo']:.3f},{pw['ci_hi']:.3f}]")

    # Strip internal _em_vals before serialising
    for r in summary_rows:
        r.pop("_em_vals", None)

    # Write outputs
    summary_csv_keys = [
        "mode", "n", "n_ok", "n_error", "em", "em_ci_lo", "em_ci_hi",
        "f1", "delta_em_vs_m1", "delta_em_ci_lo", "delta_em_ci_hi",
        "n_improved", "n_worsened", "n_changed",
        "avg_n_supported", "avg_n_unsupported", "avg_n_contradicted",
        "avg_evidence", "avg_sf_recall",
        "avg_method_cost_usd", "avg_draft_cost_usd", "avg_latency_s",
        "n_error",
    ]
    _write_csv(out_dir / "summary.csv", summary_rows, summary_csv_keys)
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    pairwise_keys = ["comparison", "delta_em", "ci_lo", "ci_hi", "n"]
    _write_csv(out_dir / "pairwise.csv", pairwise_rows, pairwise_keys)
    (out_dir / "pairwise.json").write_text(
        json.dumps(pairwise_rows, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    _plot(summary_rows, out_dir / "em_by_mode.png")

    print(f"\nOutputs written to {out_dir}/")
    print(f"  summary.csv / summary.json  — {len(summary_rows)} modes")
    print(f"  pairwise.csv / pairwise.json — {len(pairwise_rows)} pairs")
    print(f"  em_by_mode.png")

    # Also add M1 baseline row to summary for reference
    m1_ci = _bootstrap_mean_ci(list(m1_em_by_eid.values()))
    m1_row = {
        "mode": "M1_baseline",
        "n": len(common_eids),
        "n_ok": sum(1 for v in m1_em_by_eid.values() if v > 0 or True),
        "n_error": 0,
        "em": m1_em_mean,
        "em_ci_lo": m1_ci[0],
        "em_ci_hi": m1_ci[1],
        "note": "M1 baseline (no retrieval, no revision)",
    }
    baseline_path = out_dir / "m1_baseline.json"
    baseline_path.write_text(json.dumps(m1_row, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
