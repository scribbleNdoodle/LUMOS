#!/usr/bin/env python3
"""
Stage 6: rebalance the answer-letter distribution.

The previous set was severely position-biased -- unseen_wiki_LASTONE.jsonl was
A:246 B:181 C:92 D:5, so always answering "A" scored 47% and D appeared in
under 1% of questions. The cause is the prompt schema itself
(ts_creation.py:444) listing "A. <option A here>" first, which nudges instruct
models to put the true answer in slot A. The original authors shipped a
rebalanced `_dist` file (A:162 B:118 C:116 D:128) but its script is missing.

`balanced` (default) assigns target gold letters round-robin over a seeded
shuffle of record order, giving exactly n//4 per letter, then picks a uniform
permutation conditioned on the gold landing in that slot. `uniform` is a plain
shuffle.

Usage:
  python rebalance_options.py --in work/70_intersect/unseen_3way.jsonl \
      --out work/80_rebalance/unseen_wiki_2026.jsonl
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcq_common as mc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = ("A", "B", "C", "D")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "work/70_intersect/unseen_3way.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "work/80_rebalance/unseen_wiki_2026.jsonl"))
    ap.add_argument("--stats_out", default=os.path.join(HERE, "work/80_rebalance/rebalance_stats.json"))
    ap.add_argument("--mode", choices=["balanced", "uniform"], default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = list(mc.jsonl_iter(args.inp))

    before, after = Counter(), Counter()
    parsed, skipped = [], []

    for rec in records:
        try:
            qa = json.loads(rec["task_sample"])
        except (KeyError, ValueError):
            skipped.append((rec.get("meta", {}).get("src_idx"), "unparsable task_sample"))
            continue
        if not mc.options_well_formed(qa):
            skipped.append((rec.get("meta", {}).get("src_idx"), "options not A./B./C./D."))
            continue
        texts = [mc.split_option(o)[1] for o in qa["options"]]
        gold_i = LETTERS.index(qa["answer"])
        before[qa["answer"]] += 1
        parsed.append((rec, qa, texts, gold_i))

    # Target gold slot per record.
    order = list(range(len(parsed)))
    rng.shuffle(order)
    targets = [0] * len(parsed)
    if args.mode == "balanced":
        for rank, pos in enumerate(order):
            targets[pos] = rank % 4
    else:
        for pos in order:
            targets[pos] = rng.randrange(4)

    out = []
    for (rec, qa, texts, gold_i), target in zip(parsed, targets):
        distractors = [t for i, t in enumerate(texts) if i != gold_i]
        rng.shuffle(distractors)
        new_texts = []
        it = iter(distractors)
        for slot in range(4):
            new_texts.append(texts[gold_i] if slot == target else next(it))

        assert sorted(new_texts) == sorted(texts), "option text multiset changed"

        new_qa = {
            "question": qa["question"],
            "options": [f"{LETTERS[i]}. {t}" for i, t in enumerate(new_texts)],
            "answer": LETTERS[target],
        }
        after[new_qa["answer"]] += 1
        new_rec = dict(rec)
        new_rec["task_sample"] = json.dumps(new_qa, ensure_ascii=False)
        out.append(new_rec)

    mc.jsonl_write_atomic(args.out, out)

    stats = {
        "input": args.inp,
        "mode": args.mode,
        "seed": args.seed,
        "n_in": len(records),
        "n_out": len(out),
        "skipped": skipped,
        "letters_before": dict(before),
        "letters_after": dict(after),
    }
    mc.dump_json(args.stats_out, stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "skipped"}, indent=2))
    if skipped:
        print(f"[warn] skipped {len(skipped)} records (see {args.stats_out})")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
