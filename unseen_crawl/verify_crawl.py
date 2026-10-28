#!/usr/bin/env python3
"""
Sanity-check the crawled corpus against the previous one.

The load-bearing check is the spacing canary. The previous corpus was produced
with li.get_text(separator=" "), which leaves a space before punctuation that
followed a wikilink ("led by AFRICOM ."). ~81% of its lines show it. If that
share collapses toward zero, the separator was lost and the new corpus is
stylistically different from the one CRAFT's prompts were tuned on -- even
though nothing would error.

Usage:
  python verify_crawl.py --new work/00_corpus/wiki_event_only.jsonl
"""

import argparse
import json
import os
import re
import statistics
import sys

REF = ("/home/jiyoung/gayoung/CRAFT_new/assets/unseen_web/"
       "corpus_samples/wiki_event_only.jsonl")

SPACE_PUNCT = re.compile(r"\s[.,]")
TRAILING_SOURCE = re.compile(r"\([^()]*\)\s*$")


def load_texts(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["text"])
    return out


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def describe(texts, label):
    lens = [len(t) for t in texts]
    stats = {
        "label": label,
        "n": len(texts),
        "min": min(lens) if lens else 0,
        "p5": pct(lens, 5),
        "median": int(statistics.median(lens)) if lens else 0,
        "p95": pct(lens, 95),
        "max": max(lens) if lens else 0,
        "mean": round(statistics.mean(lens), 1) if lens else 0,
        "space_punct_share": round(sum(bool(SPACE_PUNCT.search(t)) for t in texts) / len(texts), 4) if texts else 0,
        "trailing_source_share": round(sum(bool(TRAILING_SOURCE.search(t)) for t in texts) / len(texts), 4) if texts else 0,
        "exact_dupes": len(texts) - len(set(texts)),
        "has_newline": sum("\n" in t for t in texts),
        "has_double_space": sum("  " in t for t in texts),
        "untrimmed": sum(t != t.strip() for t in texts),
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--meta", default="")
    ap.add_argument("--overlap-end", default="2026-01-23")
    args = ap.parse_args()

    new = load_texts(args.new)
    new_stats = describe(new, "new")
    print(json.dumps(new_stats, indent=2))

    failures = []
    if new_stats["exact_dupes"]:
        failures.append(f"{new_stats['exact_dupes']} exact duplicate bullets")
    for key in ("has_newline", "has_double_space", "untrimmed"):
        if new_stats[key]:
            failures.append(f"{new_stats[key]} lines with {key}")
    if new_stats["min"] < 40:
        failures.append(f"shortest bullet is {new_stats['min']} chars (<40)")

    if os.path.exists(args.ref):
        ref = load_texts(args.ref)
        ref_stats = describe(ref, "reference")
        print(json.dumps(ref_stats, indent=2))

        # Spacing canary.
        delta = abs(new_stats["space_punct_share"] - ref_stats["space_punct_share"])
        if delta > 0.10:
            failures.append(
                f"spacing canary off: new {new_stats['space_punct_share']:.1%} "
                f"vs reference {ref_stats['space_punct_share']:.1%} -- "
                f"separator=' ' may have been lost")

        if new_stats["trailing_source_share"] < 0.95:
            failures.append(
                f"only {new_stats['trailing_source_share']:.1%} of bullets end "
                f"in a (Source) -- reference is {ref_stats['trailing_source_share']:.1%}")

        # Overlap on the window the reference covers.
        if args.meta and os.path.exists(args.meta):
            window = set()
            with open(args.meta, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = json.loads(line)
                    if m["date"] <= args.overlap_end:
                        window.add(m["text"])
            refset = set(ref)
            inter = window & refset
            if window:
                print(f"\noverlap window (<= {args.overlap_end}): {len(window)} bullets")
                print(f"  recall vs reference : {len(inter) / len(refset):.1%}")
                print(f"  precision vs ref    : {len(inter) / len(window):.1%}")
                if len(inter) / len(window) < 0.95:
                    failures.append(
                        f"only {len(inter) / len(window):.1%} of overlap-window "
                        f"bullets appear in the reference")

    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nOK: corpus matches the reference corpus' shape")


if __name__ == "__main__":
    main()
