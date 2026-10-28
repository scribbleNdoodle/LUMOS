"""
CUDA_VISIBLE_DEVICES=0 python external/normP.py \
  --model_size 7B \
  --prompt_id 1 \
  --input_jsonl /mnt/PK/Inside-out/7B/internal_prompt1/final/AH_rebuttal_new/AH_rebuttal_new_labeling.jsonl

"""

import os
import re
import argparse
import json
import math
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

########################################
# argparse
########################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_size",
    type=str,
    required=True,
    choices=["7B", "13B", "32B"],
)
parser.add_argument(
    "--prompt_id",
    type=int,
    required=True,
    choices=[1, 2, 3, 4],
)
parser.add_argument(
    "--input_jsonl",
    type=str,
    required=True,
)

args = parser.parse_args()

########################################
# Model settings
########################################

MODEL_PATHS = {
    "7B": "/mnt/PK/models/OLMo2/OLMo-2-1124-7B-Instruct",
    "13B": "/mnt/PK/models/OLMo2/OLMo-2-1124-13B-Instruct",
    "32B": "/mnt/PK/models/OLMo2/OLMo-2-0325-32B-Instruct",
}

MODEL_PATH = MODEL_PATHS[args.model_size]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

########################################
# output path
########################################

BASE_PREFIX = "/mnt/PK"
_tag_match = re.search(r"/internal_prompt\d+(_\w+)?/", args.input_jsonl)
_tag = _tag_match.group(1) or "" if _tag_match else ""
OLD_MIDDLE = f"Inside-out/{args.model_size}/internal_prompt{args.prompt_id}{_tag}"
NEW_MIDDLE = f"Inside-out/{args.model_size}/external_prompt{args.prompt_id}{_tag}"

def build_output_path(input_path: str) -> str:
    expected_prefix = os.path.join(BASE_PREFIX, OLD_MIDDLE)
    if not input_path.startswith(expected_prefix):
        raise ValueError(f"INPUT_JSONL must start with {expected_prefix}")
    tail = os.path.relpath(input_path, expected_prefix)
    return os.path.join(BASE_PREFIX, NEW_MIDDLE, tail)

OUTPUT_JSONL = build_output_path(args.input_jsonl)

########################################
# Model loading
########################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

########################################
# Prompt builders
########################################

def build_sequence_prompt1(question, choices, answer_text):
    choice_block = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return (
        "<|endoftext|><|user|>\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{choice_block}\n"
        "<|assistant|>\n"
        f"{answer_text}\n"
        "<|endoftext|>"
    )

def build_sequence_prompt3(question, choices, answer_text, reasoning):
    choice_block = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return (
        "<|endoftext|><|user|>\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{choice_block}\n"
        "<|assistant|>\n"
        f"{reasoning}\n\n"
        "### FINAL ANSWER:\n"
        f"{answer_text}\n"
        "<|endoftext|>"
    )

########################################
# token-level log prob
########################################

@torch.no_grad()
def compute_token_logprobs(sequence, answer_text):
    enc = tokenizer(sequence, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)

    ans_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    full_ids = input_ids[0].tolist()

    start_idx = None
    for i in range(len(full_ids)):
        if full_ids[i : i + len(ans_ids)] == ans_ids:
            start_idx = i
            break

    if start_idx is None:
        return None

    logits = model(input_ids).logits
    log_probs = [
        torch.log_softmax(logits[0, start_idx + i - 1], dim=-1)[tok].item()
        for i, tok in enumerate(ans_ids)
    ]
    return log_probs


def compute_P_and_Pnorm(log_probs):
    log_P = sum(log_probs)
    return math.exp(log_P), math.exp(log_P / len(log_probs))

########################################
# Process JSONL
########################################

def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def process_jsonl():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    total_lines = count_lines(args.input_jsonl)

    with open(args.input_jsonl, "r") as fin, open(OUTPUT_JSONL, "w") as fout:
        for line in tqdm(
            fin,
            total=total_lines,
            desc="📘 Processing MCQ",
            dynamic_ncols=True
        ):
            ex = json.loads(line)

            question = ex["question"]
            choices = ex["choices"]
            answer_index = ex.get("answer")
            reasoning = ex.get("raw_samples", "")

            # Initialize default value to 0
            token_count = 0
            P = 0.0
            P_norm = 0.0

            if answer_index is not None:
                answer_text = choices[answer_index]

                if args.prompt_id in {1, 2}:
                    sequence = build_sequence_prompt1(question, choices, answer_text)
                else:
                    sequence = build_sequence_prompt3(
                        question, choices, answer_text, reasoning
                    )

                log_probs = compute_token_logprobs(sequence, answer_text)

                if log_probs is not None:
                    token_count = len(log_probs)
                    P, P_norm = compute_P_and_Pnorm(log_probs)

            ex.update(
                {
                    "prompt_id": args.prompt_id,
                    "token_count": token_count,
                    "P(a|q)": P,
                    "P_norm(a|q)": P_norm,
                }
            )

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")


########################################
# main
########################################

if __name__ == "__main__":
    start = time.time()
    process_jsonl()
    print(f"✅ Done in {(time.time() - start) / 60:.2f} min")
    print(f"Saved to: {OUTPUT_JSONL}")
