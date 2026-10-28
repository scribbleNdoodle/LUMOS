#!/usr/bin/env python3
"""
Stage 2: open-ended (no options) answering, to find what the model does NOT know.

Port of /mnt/PK/dataset/current_events/unseen_wiki/CA_current_wiki.py.

Kept verbatim: the prompt (:54-70), gold resolution (:90-121, answer letter ->
option text with the "A. " prefix stripped), sampling budget (1 greedy + 10 at
T=1.0, 15 new tokens), candidate aggregation and gold injection (:123-159).

Changed, each for a reason:
  * CUDA_VISIBLE_DEVICES="4,5,6,7" (:3) -> --gpu. Those GPUs belong to another
    job on this box; running the original unmodified would OOM it.
  * id was f"{datetime.now():%Y%m%d}_{i}" (:170), which differs per run day and
    would break the 3-model intersection -> deterministic "q{src_idx:06d}".
  * HF generate -> vLLM. The original ran 11 sequential batch-1 generate calls
    with device_map="auto"; that is ~3.5 h/model here vs ~5 min batched.
  * top_k defaults to 50, not vLLM's 0. HF's GenerationConfig default is 50 and
    OLMo-2 does not override it, so the original sampled at top_k=50. Leaving
    vLLM's default would silently change which questions count as missed.
  * bfloat16 instead of float16 (:38), matching the checkpoints.

Usage:
  python answer_openended.py --model_size 7B \
      --model_path /mnt/PK/models/OLMo2/OLMo-2-1124-7B-Instruct \
      --input work/30_dedup/mcq_clean.jsonl \
      --output work/40_answers/cands_7B.jsonl --gpu 0
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcq_common as mc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Shorthands for the OLMo-2 sizes used by the LUMOS eval. Any other model works
# too -- pass --model_label <name> --model_path <dir>.
MODEL_PATHS = {
    "7B": "/mnt/PK/models/OLMo2/OLMo-2-1124-7B-Instruct",
    "13B": "/mnt/PK/models/OLMo2/OLMo-2-1124-13B-Instruct",
    "32B": "/mnt/PK/models/OLMo2/OLMo-2-0325-32B-Instruct",
}

SYSTEM_PROMPT = "You are a concise assistant. Answer the question directly."
USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Answer concisely using a single word, a number, or a short phrase. "
    "Do not answer in a complete sentence."
)


def normalize_answer(ans: str) -> str:
    """CA_current_wiki.py:48-52."""
    return re.sub(r"\s+", " ", (ans or "").strip())


def parse_gold_answer(task_sample_str: str):
    """CA_current_wiki.py:90-121."""
    try:
        data = json.loads(task_sample_str)
    except (TypeError, ValueError):
        return None, None

    question = data.get("question", "")
    options = data.get("options", [])
    answer_key = data.get("answer", "")

    idx_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    if answer_key in idx_map and len(options) > idx_map[answer_key]:
        raw_option = options[idx_map[answer_key]]
        gold_text = re.sub(r"^[A-Z][\.\)\s]+\s*", "", raw_option)
    else:
        gold_text = str(answer_key)
    return question, gold_text


def build_messages(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]


def build_candidates(greedy_ans, sampled_answers, gold_answer):
    """CA_current_wiki.py:123-159."""
    pair_counter = Counter()
    pair_counter[(normalize_answer(greedy_ans), "greedy")] += 1
    for s in sampled_answers:
        pair_counter[(normalize_answer(s), "sampling")] += 1

    candidates = [{"answer": a, "count": c, "source": src}
                  for (a, src), c in pair_counter.items()]

    norm_gold = normalize_answer(gold_answer)
    if not any(c["answer"] == norm_gold for c in candidates):
        candidates.append({"answer": norm_gold, "count": 1, "source": "gold"})
    return candidates


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(HERE, "work/30_dedup/mcq_clean.jsonl"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_label", "--model_size", dest="model_label", required=True,
                    help="free-form name used in output filenames; 7B/13B/32B "
                         "resolve to the OLMo-2 paths automatically")
    ap.add_argument("--model_path", default="",
                    help="required unless --model_label is 7B/13B/32B")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--id_prefix", default="q")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_path = args.model_path or MODEL_PATHS.get(args.model_label, "")
    if not model_path:
        sys.exit(f"[ERROR] --model_path is required for label {args.model_label!r} "
                 f"(only {sorted(MODEL_PATHS)} are built in)")
    if not os.path.isdir(model_path):
        sys.exit(f"[ERROR] model path not found: {model_path}")

    rows = []
    for rec in mc.jsonl_iter(args.input):
        meta = rec.get("meta") or {}
        question, gold = parse_gold_answer(rec.get("task_sample"))
        if not question or gold is None:
            continue
        rows.append({
            "id": f"{args.id_prefix}{int(meta.get('src_idx', len(rows))):06d}",
            "src_idx": meta.get("src_idx"),
            "question": question,
            "gold_answer": gold,
        })
    if args.limit:
        rows = rows[:args.limit]
    print(f"[input] {len(rows)} questions from {args.input}")
    if not rows:
        print("[done] nothing to do")
        return

    from vllm import LLM, SamplingParams

    print(f"[vllm] loading {model_path} (tp={args.tp}, gpus={args.gpu})")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=False,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )

    convs = [build_messages(r["question"]) for r in rows]

    t0 = time.time()
    print("[pass 1/2] greedy")
    greedy_sp = SamplingParams(temperature=0.0, n=1, max_tokens=args.max_new_tokens)
    greedy_out = llm.chat(convs, greedy_sp, add_generation_prompt=True)

    print(f"[pass 2/2] sampling n={args.num_samples} (T={args.temperature}, "
          f"top_p={args.top_p}, top_k={args.top_k})")
    sample_sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        n=args.num_samples, max_tokens=args.max_new_tokens, seed=args.seed,
    )
    sample_out = llm.chat(convs, sample_sp, add_generation_prompt=True)

    records = []
    for r, g, s in zip(rows, greedy_out, sample_out):
        greedy_ans = g.outputs[0].text.strip()
        sampled = [o.text.strip() for o in s.outputs]
        records.append({
            "id": r["id"],
            "src_idx": r["src_idx"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "candidates": build_candidates(greedy_ans, sampled, r["gold_answer"]),
        })

    mc.jsonl_write_atomic(args.output, records)
    print(f"\n[done] {len(records)} records in {round(time.time() - t0, 1)}s "
          f"-> {args.output}")


if __name__ == "__main__":
    main()
