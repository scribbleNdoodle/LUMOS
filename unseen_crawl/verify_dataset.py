#!/usr/bin/env python3
"""
Contract check for the final dataset, against what lumos actually reads.

cA_mcq.py:486-493 (the non---realtimeQA branch):
    task = json.loads(ex["task_sample"])
    question, choices = task["question"], task["options"]
    gold_index = ord(task["answer"].upper()) - ord("A")
    qid = f"{ex.get('entity','unknown')}_{line_idx}"

So `task_sample` must be a JSON *string*, options must be a 4-element list, and
answer must be a single A-D letter. `entity: null` is intentional -- it yields
qid "None_<line_idx>", which stays unique because the line index disambiguates
(same as the previous unseen_wiki_LASTONE_dist.jsonl).

Usage:
  python verify_dataset.py --in work/80_rebalance/unseen_wiki_2026.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcq_common as mc  # noqa: E402

LETTERS = ("A", "B", "C", "D")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--expect_n", type=int, default=0)
    ap.add_argument("--letter_tol", type=float, default=0.03)
    ap.add_argument("--require_grounding", choices=["on", "off"], default="on")
    args = ap.parse_args()

    failures = []
    letters = Counter()
    src_idxs, questions = [], []
    n = 0

    for i, rec in enumerate(mc.jsonl_iter(args.inp)):
        n += 1
        where = f"line {i + 1}"

        ts = rec.get("task_sample")
        if not isinstance(ts, str):
            failures.append(f"{where}: task_sample is {type(ts).__name__}, must be str")
            continue
        try:
            qa = json.loads(ts)
        except ValueError as e:
            failures.append(f"{where}: task_sample not valid JSON ({e})")
            continue

        for key in ("question", "options", "answer"):
            if key not in qa:
                failures.append(f"{where}: task_sample missing {key!r}")
        if any(k not in qa for k in ("question", "options", "answer")):
            continue

        opts = qa["options"]
        if not isinstance(opts, list) or len(opts) != 4:
            failures.append(f"{where}: options is not a 4-element list")
            continue
        if not mc.options_well_formed(qa):
            failures.append(f"{where}: options are not prefixed A./B./C./D. in order")
            continue

        ans = qa["answer"]
        if ans not in LETTERS:
            failures.append(f"{where}: answer {ans!r} not in A-D")
            continue
        letters[ans] += 1

        if "entity" not in rec or rec["entity"] is not None:
            failures.append(f"{where}: entity should be null")

        meta = rec.get("meta") or {}
        if args.require_grounding == "on":
            src = meta.get("source_text") or ""
            gold = mc.answer_text_from_options(qa)
            if gold not in src:
                failures.append(f"{where}: gold text not verbatim in meta.source_text")

        if "src_idx" in meta:
            src_idxs.append(meta["src_idx"])
        questions.append(qa["question"])

    if len(set(src_idxs)) != len(src_idxs):
        dupes = len(src_idxs) - len(set(src_idxs))
        failures.append(f"{dupes} duplicate src_idx values")
    if len(set(questions)) != len(questions):
        dupes = len(questions) - len(set(questions))
        failures.append(f"{dupes} duplicate questions")

    if n and letters:
        for letter in LETTERS:
            share = letters.get(letter, 0) / n
            if abs(share - 0.25) > args.letter_tol:
                failures.append(
                    f"answer letter {letter} at {share:.1%}, "
                    f"outside 25%+-{args.letter_tol:.0%}")

    if args.expect_n and n != args.expect_n:
        failures.append(f"line count {n} != expected {args.expect_n}")

    print(f"records       : {n}")
    print(f"answer letters: {dict(letters)}")
    print(f"unique qs     : {len(set(questions))}")

    if failures:
        print(f"\nFAIL ({len(failures)} problems):")
        for f in failures[:40]:
            print(f"  - {f}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        sys.exit(1)

    print("\nOK: dataset satisfies the cA_mcq.py contract")


if __name__ == "__main__":
    main()
