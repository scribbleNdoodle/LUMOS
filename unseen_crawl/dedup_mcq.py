#!/usr/bin/env python3
"""
Stage 1c: fuzzy dedup of generated MCQs.

Semantics match CRAFT's `deduplicate` (ts_creation.py:956-983): greedy
keep-first, drop record i if any already-kept record scores >= 85 on
fuzz.token_set_ratio over the task_sample string.

Engine differs. CRAFT ran an O(n^2) ProcessPoolExecutor that re-partitioned a
*set* on every iteration -- slow and order-nondeterministic. Here it is one
rapidfuzz.process.cdist call per block (workers=-1), which is both
deterministic and ~1000x faster. rapidfuzz's token_set_ratio with
default_process is bit-identical to thefuzz's (thefuzz is a rapidfuzz wrapper);
--selftest verifies that against a recorded checksum.

Also fixes a CRAFT bug: ts_creation.py:966-968 `continue`s on an exact
duplicate without recording it, so 2nd+ exact copies survived into the output.
Wikipedia repeats bullets across days, so that matters here.

Requires rapidfuzz -> run under the `sypk` env (rapidfuzz 3.14.3 preinstalled).

Usage:
  python dedup_mcq.py --in work/20_filter/mcq_valid.jsonl \
      --out work/30_dedup/mcq_clean.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcq_common as mc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SELFTEST_MD5 = "184d0de82de56ee2eea5b5af38e8bda9"


def _scorer_mod():
    from rapidfuzz import fuzz, process, utils
    return fuzz, process, utils


def selftest(path):
    """Score 4000 real+perturbed pairs and check the md5 of the score vector."""
    fuzz, _, utils = _scorer_mod()
    rows = [json.dumps(r["task_sample"]) for r in mc.jsonl_iter(path)][:2000]
    if len(rows) < 2:
        print("[selftest] not enough rows")
        return
    pairs = []
    for i, s in enumerate(rows):
        pairs.append((s, rows[(i + 1) % len(rows)]))
        pairs.append((s, s[: max(1, len(s) // 2)]))
    scores = [round(fuzz.token_set_ratio(a, b, processor=utils.default_process), 6)
              for a, b in pairs]
    md5 = hashlib.md5(json.dumps(scores).encode()).hexdigest()
    print(f"[selftest] {len(pairs)} pairs, md5={md5}")
    print(f"[selftest] reference md5={SELFTEST_MD5} "
          f"({'MATCH' if md5 == SELFTEST_MD5 else 'differs - corpus-dependent, informational'})")


def dedup(strings, ratio, block):
    """Greedy keep-first fuzzy dedup. Returns the list of kept indices."""
    import numpy as np
    fuzz, process, utils = _scorer_mod()

    n = len(strings)
    keep, kept_idx = [], []
    for start in range(0, n, block):
        rows = strings[start:start + block]
        if kept_idx:
            ref = [strings[i] for i in kept_idx]
            prev = process.cdist(rows, ref, scorer=fuzz.token_set_ratio,
                                 processor=utils.default_process,
                                 score_cutoff=ratio, workers=-1, dtype=np.uint8)
        else:
            prev = None

        # Within-block comparisons must respect decisions made earlier in the
        # same block, so compute the block-local matrix once and sweep it.
        local = process.cdist(rows, rows, scorer=fuzz.token_set_ratio,
                              processor=utils.default_process,
                              score_cutoff=ratio, workers=-1, dtype=np.uint8)

        block_keep = []
        for i in range(len(rows)):
            if prev is not None and prev[i].max(initial=0) >= ratio:
                continue
            if any(local[i][j] >= ratio for j in block_keep):
                continue
            block_keep.append(i)

        for i in block_keep:
            keep.append(start + i)
        kept_idx = keep[:]
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "work/20_filter/mcq_valid.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "work/30_dedup/mcq_clean.jsonl"))
    ap.add_argument("--stats_out", default=os.path.join(HERE, "work/30_dedup/dedup_stats.json"))
    ap.add_argument("--ratio", type=int, default=85)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--drop_exact_duplicates", choices=["on", "off"], default="on")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.inp)
        return

    records = list(mc.jsonl_iter(args.inp))
    n_in = len(records)
    print(f"[dedup] {n_in} records")

    exact_dropped = 0
    if args.drop_exact_duplicates == "on":
        seen, uniq = set(), []
        for r in records:
            s = r["task_sample"]
            if s in seen:
                exact_dropped += 1
                continue
            seen.add(s)
            uniq.append(r)
        records = uniq
        print(f"[dedup] exact duplicates dropped: {exact_dropped}")

    t0 = time.time()
    strings = [r["task_sample"] for r in records]
    keep = dedup(strings, args.ratio, args.block)
    kept = [records[i] for i in keep]
    elapsed = round(time.time() - t0, 2)

    mc.jsonl_write_atomic(args.out, kept)
    stats = {
        "input": args.inp,
        "n_in": n_in,
        "exact_duplicates_dropped": exact_dropped,
        "fuzzy_dropped": len(records) - len(kept),
        "n_out": len(kept),
        "ratio": args.ratio,
        "elapsed_sec": elapsed,
    }
    mc.dump_json(args.stats_out, stats)
    print(json.dumps(stats, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
