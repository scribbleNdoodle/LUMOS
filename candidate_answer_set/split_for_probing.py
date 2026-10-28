"""
python split_for_probing.py \
  --input_jsonl /mnt/PK/Inside-out/32B/candidate_answer_set_prompt3/final/AT_rebuttal_new.jsonl \
  --train_ratio 0.25

"""

"""
Split labeled JSONL into Train / Dev by Question ID (no leakage).
- Input JSONL already contains `label`
- Split by base question ID
"""

import json
import os
import random
import argparse
import re
from collections import defaultdict
from typing import List, Dict


# -------------------------
# utils
# -------------------------
def extract_qid(example_id: str) -> str:
    """
    Extract base question ID.
    e.g. "20250328_0_0" -> "20250328_0"
    """
    if "_" not in example_id:
        return example_id
    return example_id.rsplit("_", 1)[0]


def build_output_paths(input_jsonl: str):
    """
    candidate_answer_set_promptX → internal_promptX
    *_ABCD.jsonl → *_probing.jsonl / *_labeling.jsonl
    """

    m = re.search(r"/candidate_answer_set_prompt(\d+)(_\w+)?/", input_jsonl)
    if not m:
        raise ValueError(
            "input_jsonl must contain /candidate_answer_set_promptX/ or /candidate_answer_set_promptX_tag/"
        )

    prompt_id = m.group(1)
    tag = m.group(2) or ""

    base_path1 = input_jsonl.replace(
        f"/candidate_answer_set_prompt{prompt_id}{tag}/",
        f"/internal_prompt{prompt_id}{tag}/"
    )
    base_path = base_path1.replace(
        f"/final/",
        f"/final/AT_rebuttal_new/"
    )

    dir_path = os.path.dirname(base_path)
    base = os.path.basename(base_path)
    name, ext = os.path.splitext(base)

    if ext != ".jsonl":
        raise ValueError("input_jsonl must be a .jsonl file")

    train_path = os.path.join(dir_path, f"{name}_probing.jsonl")
    dev_path = os.path.join(dir_path, f"{name}_labeling.jsonl")

    return train_path, dev_path


# -------------------------
# argparse
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Split JSONL into probing (train) / labeling (dev) by question ID"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input labeled jsonl file"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.25,
        help="Ratio of questions used for probing (default: 0.25)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()


# -------------------------
# main
# -------------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    OUTPUT_TRAIN_JSONL, OUTPUT_DEV_JSONL = build_output_paths(args.input_jsonl)

    # -----------------------
    # 1. Load data
    # -----------------------
    print(f"Loading labeled data from {args.input_jsonl}...")
    all_rows: List[Dict] = []

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_rows.append(json.loads(line))

    print(f"Loaded {len(all_rows)} rows")

    # -----------------------
    # 2. Group by Question ID
    # -----------------------
    by_qid = defaultdict(list)

    for row in all_rows:
        qid = extract_qid(row["id"])
        by_qid[qid].append(row)

    all_qids = list(by_qid.keys())
    random.shuffle(all_qids)

    # -----------------------
    # 3. Split
    # -----------------------
    split_idx = int(len(all_qids) * args.train_ratio)
    train_qids = set(all_qids[:split_idx])
    dev_qids = set(all_qids[split_idx:])

    print(f"Total questions: {len(all_qids)}")
    print(f"Train questions: {len(train_qids)} ({args.train_ratio*100:.1f}%)")
    print(f"Dev   questions: {len(dev_qids)} ({(1-args.train_ratio)*100:.1f}%)")

    # -----------------------
    # 4. Save files
    # -----------------------
    os.makedirs(os.path.dirname(OUTPUT_TRAIN_JSONL), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_DEV_JSONL), exist_ok=True)

    def write_split(path, qid_set):
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for qid in all_qids:  # keep shuffled order
                if qid in qid_set:
                    for row in by_qid[qid]:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n += 1
        return n

    n_train = write_split(OUTPUT_TRAIN_JSONL, train_qids)
    n_dev = write_split(OUTPUT_DEV_JSONL, dev_qids)

    print("Done.")
    print(f"Train saved to {OUTPUT_TRAIN_JSONL} ({n_train} rows)")
    print(f"Dev   saved to {OUTPUT_DEV_JSONL} ({n_dev} rows)")


if __name__ == "__main__":
    main()
