#!/usr/bin/env python3
"""
Stage 1b: parse raw generations and apply the grounding filter.

Port of create_task_samples_diff.py:356-380.

The grounding filter is the real quality gate: the correct option's text must
appear verbatim (case-sensitive) in the source bullet. On the previous run it
took 4354 raw -> 888 clean (20.4%).

--relax_span stays "none" (the original rule). The stats file reports how many
*additional* records a looser rule would have kept, so the decision to relax is
data-driven rather than a guess.

Usage:
  python filter_mcq.py --in work/10_gen/mcq_raw.jsonl \
      --out work/20_filter/mcq_valid.jsonl
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcq_common as mc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "work/10_gen/mcq_raw.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "work/20_filter/mcq_valid.jsonl"))
    ap.add_argument("--errors_csv", default=os.path.join(HERE, "work/20_filter/mcq_errors.csv"))
    ap.add_argument("--stats_out", default=os.path.join(HERE, "work/20_filter/filter_stats.json"))
    ap.add_argument("--relax_span", choices=["none", "case", "ws"], default="none")
    args = ap.parse_args()

    kept, errors = [], []
    parse_errors = Counter()
    n = 0
    span_drops = 0
    malformed_options = 0
    would_pass = Counter()
    letters = Counter()

    for rec in mc.jsonl_iter(args.inp):
        n += 1
        meta = rec.get("meta") or {}
        source_text = (meta.get("source_text") or "").strip()

        try:
            qa = mc.qa_mc_extract(rec.get("task_sample") or "")
        except Exception as e:  # noqa: BLE001 - mirrors the original catch-all
            parse_errors[type(e).__name__] += 1
            errors.append((meta.get("src_idx"), f"{type(e).__name__}: {e}"))
            continue

        if not mc.options_well_formed(qa):
            # CRAFT's _opt_text split on the first ". " and silently mangled
            # these; we drop and log them instead.
            malformed_options += 1
            errors.append((meta.get("src_idx"), "MalformedOptions: not A./B./C./D."))
            continue

        if not mc.passes_answer_span_filter(qa, source_text, relax=args.relax_span):
            span_drops += 1
            for mode in ("case", "ws"):
                if mc.passes_answer_span_filter(qa, source_text, relax=mode):
                    would_pass[mode] += 1
            continue

        letters[qa["answer"]] += 1
        kept.append(mc.clean_record(qa, meta))

    mc.jsonl_write_atomic(args.out, kept)

    os.makedirs(os.path.dirname(os.path.abspath(args.errors_csv)), exist_ok=True)
    with open(args.errors_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_idx", "exception"])
        w.writerows(errors)

    stats = {
        "input": args.inp,
        "n_raw": n,
        "n_kept": len(kept),
        "pass_rate": round(len(kept) / n, 4) if n else 0.0,
        "parse_errors": dict(parse_errors),
        "malformed_options": malformed_options,
        "span_filter_drops": span_drops,
        "relax_span": args.relax_span,
        "extra_if_relaxed": dict(would_pass),
        "answer_letters": dict(letters),
    }
    mc.dump_json(args.stats_out, stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
