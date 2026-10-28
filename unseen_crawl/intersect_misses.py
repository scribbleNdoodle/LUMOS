#!/usr/bin/env python3
"""
Stage 5: keep only the questions ALL THREE OLMo-2 sizes missed.

The previous pipeline used 32B alone (최종추출.py) and joined on question
*text*, because the source file had no id. Stage 2 now emits a deterministic
src_idx, so this joins on that instead.

  keep = misses(7B) ∩ misses(13B) ∩ misses(32B)

then re-selects those records from the Stage-1 clean file so the output carries
the exact downstream contract ({task_sample, entity, chunks, meta}).

Measured on the previous data: 7B missed 618, 13B 556, 32B 524, and the 3-way
intersection was 452/878 = 51.5% -- only 72 fewer than 32B alone.

Usage:
  python intersect_misses.py --misses work/60_split/misses_7B.jsonl \
      work/60_split/misses_13B.jsonl work/60_split/misses_32B.jsonl \
      --source work/30_dedup/mcq_clean.jsonl \
      --out work/70_intersect/unseen_3way.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcq_common as mc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_keys(path, join_key):
    keys, n = set(), 0
    for rec in mc.jsonl_iter(path):
        n += 1
        if join_key == "src_idx":
            k = rec.get("src_idx")
            if k is None:
                raise SystemExit(
                    f"{path}: record without src_idx -- regenerate Stage 2, or "
                    f"use --join_key question")
        else:
            k = rec.get("question")
        keys.add(k)
    return keys, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--misses", nargs="+", required=True,
                    help="miss files, one per model")
    ap.add_argument("--labels", nargs="*", default=[],
                    help="labels matching --misses (default: basenames)")
    ap.add_argument("--source", default=os.path.join(HERE, "work/30_dedup/mcq_clean.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "work/70_intersect/unseen_3way.jsonl"))
    ap.add_argument("--venn_out", default=os.path.join(HERE, "work/70_intersect/venn.json"))
    ap.add_argument("--join_key", choices=["src_idx", "question"], default="src_idx")
    ap.add_argument("--require", type=int, default=0,
                    help="keep questions missed by at least N models "
                         "(default: all of them)")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(p).replace(".jsonl", "") for p in args.misses]
    if len(labels) != len(args.misses):
        ap.error("--labels count must match --misses count")

    require = args.require or len(args.misses)
    if require > len(args.misses):
        ap.error("--require exceeds the number of miss files")

    per_model = {}
    for label, path in zip(labels, args.misses):
        keys, n = load_keys(path, args.join_key)
        per_model[label] = keys
        print(f"[{label}] {n} misses ({len(keys)} unique keys)")

    counts = {}
    for keys in per_model.values():
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
    keep_keys = {k for k, c in counts.items() if c >= require}
    print(f"[intersect] require>={require} -> {len(keep_keys)} keys")

    source_records = list(mc.jsonl_iter(args.source))
    kept, missing = [], 0
    for i, rec in enumerate(source_records):
        meta = rec.get("meta") or {}
        if args.join_key == "src_idx":
            k = meta.get("src_idx", i)
        else:
            try:
                k = json.loads(rec["task_sample"]).get("question")
            except (KeyError, ValueError):
                k = None
        if k in keep_keys:
            kept.append(rec)
    matched = {(r.get("meta") or {}).get("src_idx") for r in kept}
    if args.join_key == "src_idx":
        missing = len(keep_keys - matched)

    mc.jsonl_write_atomic(args.out, kept)

    all_keys = set().union(*per_model.values()) if per_model else set()
    venn = {
        "join_key": args.join_key,
        "require": require,
        "source_total": len(source_records),
        "per_model_misses": {k: len(v) for k, v in per_model.items()},
        "union": len(all_keys),
        "intersection": len(keep_keys),
        "kept_records": len(kept),
        "keys_without_source_record": missing,
        "pairwise": {
            f"{a}&{b}": len(per_model[a] & per_model[b])
            for i, a in enumerate(labels) for b in labels[i + 1:]
        },
        "miss_rate_per_model": {
            k: round(len(v) / len(source_records), 4) if source_records else 0.0
            for k, v in per_model.items()
        },
        "intersection_rate": round(len(keep_keys) / len(source_records), 4) if source_records else 0.0,
    }
    mc.dump_json(args.venn_out, venn)

    if missing:
        print(f"[warn] {missing} intersected keys had no matching source record")
    print(json.dumps(venn, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
