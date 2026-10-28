#!/usr/bin/env python3
"""
Stage 1: two-turn event-only MCQ generation with vLLM.

Port of CRAFT_new/code/create_task_samples_diff.py (event branch, lines
148-352) onto vLLM 0.13 + the *correctly scoped* EVENT_ONLY prompts.

Differences from the original, each deliberate:
  * prompts come from prompts.py by name, so the getattr fallback bug
    (create_task_samples_diff.py:169-178) cannot recur -- asserted at startup
  * LLM.chat() by default; CRAFT used LLM.generate() with no chat template,
    which is out-of-distribution for an -Instruct checkpoint
  * tensor_parallel_size is a CLI arg, not torch.cuda.device_count()
    (common.py:207 would otherwise seize every visible GPU)
  * max_logprobs dropped (unused); max_model_len passed through to EngineArgs
  * meta carries src_idx -- the join key for the 3-model intersection

Usage:
  python gen_mcq.py --corpus work/00_corpus/wiki_event_only.jsonl \
      --out_raw work/10_gen/mcq_raw.jsonl --gpu 0,1 --tp 2
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcq_common as mc  # noqa: E402
import prompts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "/mnt/PK/models/Qwen/Qwen2.5-32B-Instruct"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=os.path.join(HERE, "work/00_corpus/wiki_event_only.jsonl"))
    ap.add_argument("--out_raw", default=os.path.join(HERE, "work/10_gen/mcq_raw.jsonl"))
    ap.add_argument("--meta_out", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu", default="0,1")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_prompt_tokens", type=int, default=7000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chat_template", choices=["on", "off"], default="on")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def assert_prompts_are_event_only():
    """The single check that proves the CRAFT fallback bug is not back."""
    t1 = prompts.QA_MC_INSTRUCTION_EVENT_ONLY.splitlines()[0]
    t2 = prompts.QA_MC_REVIEW_INSTRUCTION_EVENT_ONLY.splitlines()[0]
    assert t1 == prompts.TURN1_HEAD, f"turn1 prompt is not event-only: {t1!r}"
    assert t2 == prompts.TURN2_HEAD, f"turn2 prompt is not event-only: {t2!r}"
    assert t1 != prompts.GENERIC_QA_MC_INSTRUCTION_HEAD
    assert t2 != prompts.GENERIC_QA_MC_REVIEW_INSTRUCTION_HEAD
    print(f"[prompt] turn1 head: {t1}")
    print(f"[prompt] turn2 head: {t2}")


def load_corpus(path, offset, limit):
    items = []
    for i, row in enumerate(mc.jsonl_iter(path)):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        items.append({"src_idx": i, "text": text})
    if offset:
        items = items[offset:]
    if limit:
        items = items[:limit]
    return items


def done_src_idxs(path):
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            idx = (rec.get("meta") or {}).get("src_idx")
            if idx is not None:
                seen.add(idx)
    return seen


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    assert_prompts_are_event_only()

    items = load_corpus(args.corpus, args.offset, args.limit)
    print(f"[corpus] {len(items)} bullets from {args.corpus}")

    if args.resume:
        skip = done_src_idxs(args.out_raw)
        if skip:
            items = [it for it in items if it["src_idx"] not in skip]
            print(f"[resume] skipping {len(skip)} already generated, {len(items)} left")
    if not items:
        print("[done] nothing to do")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    # Prompt-length guard, replacing check_prompt_length_modified
    # (ts_creation.py:196-246) but keeping its (idx, prompt) tuple semantics.
    kept, dropped_long = [], 0
    for it in items:
        n = len(tok(mc.build_turn1_prompt(it["text"]), add_special_tokens=False)["input_ids"])
        if n > args.max_prompt_tokens:
            dropped_long += 1
            continue
        kept.append(it)
    items = kept
    if dropped_long:
        print(f"[len-filter] dropped {dropped_long} over-long prompts")

    print(f"[vllm] loading {args.model} (tp={args.tp}, gpus={args.gpu})")
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=False,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )

    use_chat = args.chat_template == "on"

    def run(batch_inputs):
        if use_chat:
            outs = llm.chat(batch_inputs, sp, add_generation_prompt=True)
        else:
            outs = llm.generate(batch_inputs, sp)
        return [o.outputs[0].text.strip() for o in outs]

    t0 = time.time()
    print(f"[turn1] generating {len(items)} drafts (chat_template={args.chat_template})")
    if use_chat:
        turn1_in = [mc.turn1_messages(it["text"]) for it in items]
    else:
        turn1_in = [mc.build_turn1_prompt(it["text"]) for it in items]
    drafts = run(turn1_in)

    print(f"[turn2] reviewing {len(items)}")
    if use_chat:
        turn2_in = [mc.turn2_messages(it["text"], d) for it, d in zip(items, drafts)]
    else:
        turn2_in = [mc.build_turn2_prompt(it["text"], d) for it, d in zip(items, drafts)]
    finals = run(turn2_in)

    records = []
    for it, draft, final in zip(items, drafts, finals):
        records.append({
            "task_sample": final,
            "draft_task_sample": draft,
            "entity": None,
            "chunks": None,
            "meta": {"source_text": it["text"], "src_idx": it["src_idx"]},
        })

    if args.resume and os.path.exists(args.out_raw):
        prev = list(mc.jsonl_iter(args.out_raw))
        records = prev + records
        records.sort(key=lambda r: r["meta"]["src_idx"])

    mc.jsonl_write_atomic(args.out_raw, records)

    elapsed = round(time.time() - t0, 1)
    meta_path = args.meta_out or os.path.join(os.path.dirname(args.out_raw), "gen_meta.json")
    mc.dump_json(meta_path, {
        "model": args.model,
        "chat_template": args.chat_template,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
                     "seed": args.seed},
        "tp": args.tp, "gpu": args.gpu,
        "corpus": args.corpus,
        "n_records": len(records),
        "dropped_long_prompts": dropped_long,
        "elapsed_sec": elapsed,
        "prompt_heads": {"turn1": prompts.TURN1_HEAD, "turn2": prompts.TURN2_HEAD},
    })

    print(f"\n[done] {len(records)} records in {elapsed}s -> {args.out_raw}")


if __name__ == "__main__":
    main()
