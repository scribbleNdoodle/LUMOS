#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VENDORED COPY -- unmodified.

source : /mnt/PK/dataset/realtimeqa_public/3.extract_realtime.py
sha256 : 98a15c332f76f4a9aa237a220858cda933aa6a592a357bce3b83242aa7f939d7
copied : 2026-07-30

Splits judged candidates into hits (the model produced the gold answer on its
own -> NOT unseen) and misses (-> unseen candidates), via
is_non_gold_hit(c) = c["label"] == 1 and c["source"] != "gold".
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def is_non_gold_hit(c: Dict[str, Any]) -> bool:
    return (c.get("label") == 1) and (str(c.get("source", "")).lower() != "gold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input jsonl (one question per line)")
    ap.add_argument("--hits_out", required=True, help="output jsonl: questions with non-gold label=1 candidates")
    ap.add_argument("--filtered_out", required=True, help="output jsonl: remove non-gold label=1 candidates from candidates")
    ap.add_argument("--misses_only_out", default="", help="(optional) output jsonl: only questions with NO non-gold hits")
    args = ap.parse_args()

    in_path = Path(args.input)
    hits_path = Path(args.hits_out)
    filtered_path = Path(args.filtered_out)
    misses_path = Path(args.misses_only_out) if args.misses_only_out else None

    total_q = 0
    q_with_hits = 0
    removed_candidates = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         hits_path.open("w", encoding="utf-8") as fhits, \
         filtered_path.open("w", encoding="utf-8") as ffiltered, \
         (misses_path.open("w", encoding="utf-8") if misses_path else open(Path("/dev/null"), "w", encoding="utf-8")) as fmiss:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_q += 1
            rec = json.loads(line)

            cands: List[Dict[str, Any]] = rec.get("candidates", []) or []
            hits = [c for c in cands if is_non_gold_hit(c)]

            if hits:
                q_with_hits += 1

                # 1) hits report
                hit_obj = {
                    "id": rec.get("id"),
                    "question": rec.get("question"),
                    "gold_answer": rec.get("gold_answer"),
                    "non_gold_hits": [
                        {
                            "answer": h.get("answer"),
                            "source": h.get("source"),
                            "count": h.get("count", 1),
                        }
                        for h in hits
                    ],
                }
                fhits.write(json.dumps(hit_obj, ensure_ascii=False) + "\n")

                # 2) filtered record: remove those candidates
                filtered_cands = [c for c in cands if not is_non_gold_hit(c)]
                removed_candidates += (len(cands) - len(filtered_cands))

                rec_filtered = dict(rec)
                rec_filtered["candidates"] = filtered_cands
                ffiltered.write(json.dumps(rec_filtered, ensure_ascii=False) + "\n")
            else:
                # no hits → 그대로 filtered에 남김 (제거할 게 없음)
                ffiltered.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # 3) misses-only (optional)
                if misses_path:
                    fmiss.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Done.\n"
        f"- total questions: {total_q}\n"
        f"- questions with non-gold label=1 hits: {q_with_hits}\n"
        f"- removed candidates (non-gold hits): {removed_candidates}\n"
        f"- hits_out: {hits_path}\n"
        f"- filtered_out: {filtered_path}\n"
        f"- misses_only_out: {misses_path if misses_path else '(not written)'}"
    )


if __name__ == "__main__":
    main()