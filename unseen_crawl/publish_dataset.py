#!/usr/bin/env python3
"""
Stage 7: publish the final dataset where lumos can read it.

cA_mcq.py:107-116 raises ValueError unless --input_jsonl starts with
/mnt/PK/MCQ or /mnt/PK/dataset, and mirrors the relative tail into the output
path. So the file has to live under one of those roots regardless of where the
code lives.

Also refuses any destination containing a path component named "final":
split_for_probing.py:51-55 rewrites "/final/" to "/final/AT_rebuttal_new/",
which would silently relocate every downstream artifact.

Usage:
  python publish_dataset.py --in work/80_rebalance/unseen_wiki_2026.jsonl \
      --dest_dir /mnt/PK/dataset/current_events/unseen_wiki_2026 \
      --stem unseen_wiki_2026
"""

import argparse
import os
import shutil
import sys

ALLOWED_ROOTS = ("/mnt/PK/MCQ", "/mnt/PK/dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--dest_dir",
                    default="/mnt/PK/dataset/current_events/unseen_wiki_2026")
    ap.add_argument("--stem", default="unseen_wiki_2026")
    args = ap.parse_args()

    dest_dir = os.path.abspath(args.dest_dir)

    if not any(dest_dir.startswith(r) for r in ALLOWED_ROOTS):
        sys.exit(f"[ERROR] dest must start with one of {ALLOWED_ROOTS} "
                 f"(cA_mcq.py:107-116 rejects anything else); got {dest_dir}")
    if "final" in dest_dir.split(os.sep):
        sys.exit("[ERROR] a path component named 'final' triggers the "
                 "split_for_probing.py:51-55 rewrite; rename it")
    if not os.path.exists(args.inp):
        sys.exit(f"[ERROR] input not found: {args.inp}")

    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{args.stem}.jsonl")
    tmp = dest + ".tmp"
    shutil.copyfile(args.inp, tmp)
    os.replace(tmp, dest)

    n = sum(1 for line in open(dest, encoding="utf-8") if line.strip())
    print(f"published {n} records")
    print(f"-> {dest}")
    print("\nRegister it with:")
    subdir = os.path.relpath(dest_dir, "/mnt/PK/dataset")
    print(f'  run_eval_unseen.sh  DATASETS += "{subdir} {args.stem}"')
    print(f"  run_cA_unseen.sh    --input_jsonl {dest}")


if __name__ == "__main__":
    main()
