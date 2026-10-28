"""
CUDA_VISIBLE_DEVICES=0 python cA_mcq.py \
  --model_size 7B \
  --input_jsonl /mnt/PK/MCQ/final_mcq/AH_rebuttal_new.jsonl \
  --prompt_id 4

"""
import os
import sys
import argparse
import json
import re
import random
import time
from typing import List, Dict
from collections import Counter, defaultdict

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
    "--input_jsonl",
    type=str,
    required=True,
)
parser.add_argument(
    "--prompt_id",
    type=int,
    required=True,
    choices=[1, 2, 3, 4],
)
parser.add_argument(
    "--realtimeQA",
    action="store_true",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Tag appended to output directory name (e.g. rebuttal → candidate_answer_set_prompt1_rebuttal)",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Ignore existing output and regenerate from scratch (default: resume)",
)

args = parser.parse_args()

########################################
# Model-specific settings
########################################

MODEL_CONFIGS = {
    "7B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-1124-7B-Instruct",
        "max_new_tokens": 1024,
    },
    "13B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-1124-13B-Instruct",
        "max_new_tokens": 1024,
    },
    "32B": {
        "model_path": "/mnt/PK/models/OLMo2/OLMo-2-0325-32B-Instruct",
        "max_new_tokens": 1024,
    },
}

cfg = MODEL_CONFIGS[args.model_size]

########################################
# Default settings
########################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = cfg["model_path"]
MAX_NEW_TOKENS = cfg["max_new_tokens"]

_tag = f"_{args.tag}" if args.tag else ""
INSERT_DIR = f"Inside-out/{args.model_size}/candidate_answer_set_prompt{args.prompt_id}{_tag}"

NUM_SAMPLES = 10
TEMPERATURE = 1.0

BASE_PREFIX = "/mnt/PK"
DATASET_DIRS = ["MCQ", "dataset"]
INPUT_JSONL = args.input_jsonl

random.seed(42)

########################################
# Output path
########################################

def build_output_path(input_path: str) -> str:
    for dataset_dir in DATASET_DIRS:
        expected_prefix = os.path.join(BASE_PREFIX, dataset_dir)
        if input_path.startswith(expected_prefix):
            relative_path = os.path.relpath(input_path, expected_prefix)
            return os.path.join(BASE_PREFIX, INSERT_DIR, relative_path)
    raise ValueError(
        f"input_jsonl must start with one of: "
        + ", ".join(os.path.join(BASE_PREFIX, d) for d in DATASET_DIRS)
    )


OUTPUT_JSONL = build_output_path(INPUT_JSONL)

########################################
# Model loading
########################################

# Loaded lazily so that an already-completed run can exit before paying the
# model-loading cost (matters for 32B).
tokenizer = None
model = None


def load_model():
    global tokenizer, model

    if model is not None:
        return

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

########################################
# Prompt
########################################

def build_prompt(question: str, choices: List[str], prompt_id: int) -> str:
    """
    prompt_id 1 : zero-shot
    prompt_id 2 : few-shot
    prompt_id 3 : zero-shot CoT
    prompt_id 4 : few-shot CoT
    """
    choice_text = ", ".join(choices)

    if prompt_id == 1:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are given a multiple-choice question. "
                    "You must answer ONLY ONE letter: A, B, C, or D. "
                    "Do not output anything else."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Choices:\n{choice_text}\n\n"
                    "Answer with only one letter (A, B, C, or D)."
                )
            }
        ]
    elif prompt_id == 2:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are given a multiple-choice question. "
                    "You must answer ONLY ONE letter: A, B, C, or D. "
                    "Do not output anything else."
                )
            },
            {
                "role": "user",
                "content": (
                    "Question: What is the relationship between Carmen Miranda and Amaro Miranda da Cunha?\n"
                    "Choices: A. Siblings, B. Aunt and nephew, C. Mother and son, D. Cousins\n"
                    "Answer: A \n\n"

                    "Question: In 2008, Martin Winter founded which significant research center at the University of Münster?\n"
                    "Choices: A. Paul Scherrer Institute, B. Institute of Chemical Technology of Inorganic Materials, "
                    "C. MEET Battery Research Center, D. Institute of Physical Chemistry\n"
                    "Answer: C \n\n"

                    "Question: What did William Lyon Mackenzie King famously say about Canada's geography and history?\n"
                    "Choices: A. Canada has too much diversity, B. Canada has too much culture, "
                    "C. Canada has too much history, D. Canada has too much geography\n"
                    "Answer: D \n\n"

                    "Question: Who is the head chef of Gordon Ramsay au Trianon?\n"
                    "Choices: A. Frederic Larquemin, B. Joël Robuchon, C. Gordon Ramsay, D. Alain Ducasse\n"
                    "Answer: A \n\n"

                    "Question: What is the architectural style of St. John in the Wilderness in Nainital?\n"
                    "Choices: A. Modern, B. Gothic, C. Baroque, D. Renaissance\n"
                    "Answer: B \n\n"

                    f"Question: {question}\n"
                    f"Choices: {choice_text}\n"
                    "Answer with only one letter (A, B, C, or D)."
                )
            }
        ]

    elif prompt_id == 3:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are given a multiple-choice question (A, B, C, D). "
                    "First, reason step by step and explain your thinking clearly. "
                    "Then, provide your final answer as only one letter: A, B, C, or D. "
                    "Always write the final answer in the format '### FINAL ANSWER: X', "
                    "where X is one of A, B, C, or D. Do not include anything else after this line."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Choices: {choice_text}\n"
                    "Write out your reasoning step by step, and then write the final answer in this format:\n"
                    "### FINAL ANSWER: "
                )
            }
        ]

    elif prompt_id == 4:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are given a multiple-choice question (A, B, C, D). "
                    "First, reason step by step and explain your thinking clearly. "
                    "Then, provide your final answer as only one letter: A, B, C, or D. "
                    "Always write the final answer in the format '### FINAL ANSWER: X', "
                    "where X is one of A, B, C, or D. Do not include anything else after this line."
                )
            },
            {
                "role": "user",
                "content": (
                    "Question: Which geographical feature is located to the east and northeast of Scottsdale, Arizona?\n"
                    "Choices: A. The Grand Canyon, B. The McDowell Mountain Range, C. The Sierra Madre Mountains, D. The Rocky Mountains\n"
                    "Reasoning: Scottsdale, Arizona is located in the Sonoran Desert. The Grand Canyon is to the northwest of Scottsdale. The Sierra Madre Mountains are located in Mexico, far to the south. The Rocky Mountains are much further to the northeast, in states like Colorado and Wyoming. The McDowell Mountain Range is a series of mountains located to the northeast of Scottsdale, Arizona. This range is a prominent geographical feature in the area and is known for its hiking trails and scenic views. Therefore, the correct answer is B. The McDowell Mountain Range."
                    "### FINAL ANSWER: B \n\n"
                    
                    f"Question: {question}\n"
                    f"Choices: {choice_text}\n"
                )
            }
        ]


    else:
        raise ValueError(prompt_id)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

########################################
# Parsing (per prompt_id)
########################################

_final_answer_re = re.compile(r"### FINAL ANSWER:\s*([ABCD])", re.IGNORECASE)
_letter_only_re = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_paren_answer_re = re.compile(r"\(\s*([ABCD])\s*\)", re.IGNORECASE)

def parse_answer(decoded: str, prompt_id: int):
    """
    decoded: model output string
    prompt_id: prompt ID

    return:
        idx: 0=A, 1=B, ... 
        letter: 'A'-'D' or None
        reasoning_text: pure CoT reasoning (prompt_id==3,4), otherwise the full decoded text
    """
    reasoning_text = decoded.strip()
    letter = None

    if prompt_id in [3,4]:
        # Extract only the letter after "FINAL ANSWER:"
        m = _final_answer_re.search(decoded)
        if m:
            letter = m.group(1).upper()
            # Remove the final answer line from reasoning_text
            reasoning_text = re.sub(r"### FINAL ANSWER:\s*[ABCD]", "", reasoning_text, flags=re.IGNORECASE).strip()

    elif prompt_id == 1:
        # greedy/sample answer extraction
        m = _letter_only_re.search(decoded)
        if m:
            letter = m.group(1).upper()

    elif prompt_id == 2:
        # prompt 2 pattern
        patterns = [
            _paren_answer_re,
            re.compile(r"answer is\s*\(?([ABCD])\)?", re.IGNORECASE),
            _letter_only_re,
        ]
        for pat in patterns:
            m = pat.search(decoded)
            if m:
                letter = m.group(1).upper()
                break

    idx = ord(letter) - ord("A") if letter else None
    return idx, letter, reasoning_text

########################################
# Generation
########################################

@torch.no_grad()
def generate_answer_index(question: str, choices: List[str], do_sample: bool):
    prompt = build_prompt(question, choices, args.prompt_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        temperature=TEMPERATURE if do_sample else None,
        top_p=1.0 if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    idx, letter, reasoning_text = parse_answer(decoded, args.prompt_id)
    return idx, letter, reasoning_text

########################################
# Generate candidates
########################################

def build_candidate_rows1(qid, question, choices, gold_index):
    counter = Counter()
    raw_store = defaultdict(list)
    letter_store = defaultdict(list)

    def record(idx, source, raw, letter):
        counter[(idx, source)] += 1
        raw_store[(idx, source)].append(raw)
        letter_store[(idx, source)].append(letter)

    idx, letter, raw = generate_answer_index(question, choices, do_sample=False)
    record(idx, "greedy", raw, letter)

    for _ in range(NUM_SAMPLES):
        idx, letter, raw = generate_answer_index(question, choices, do_sample=True)
        record(idx, "sampling", raw, letter)

    rows = []
    row_idx = 0
    for (pred, source), count in counter.items():
        rows.append({
            "id": f"{qid}_{row_idx}",
            "question": question,
            "choices": choices,
            "gold_answer": gold_index,
            "answer": pred,
            "count": count,
            "source": source,
            "label": int(pred == gold_index),
            "raw_samples": raw_store[(pred, source)],
            "raw_letters": letter_store[(pred, source)],
        })
        row_idx += 1

    rows.append({
        "id": f"{qid}_{row_idx}",
        "question": question,
        "choices": choices,
        "gold_answer": gold_index,
        "answer": gold_index,
        "count": 1,
        "source": "gold",
        "label": 1,
        "raw_samples": None,
        "raw_letters": None,
    })

    return rows

def build_candidate_rows(qid, question, choices, gold_index):
    rows = []
    row_idx = 0

    def make_row(pred, source, raw, letter):
        nonlocal row_idx
        rows.append({
            "id": f"{qid}_{row_idx}",
            "question": question,
            "choices": choices,
            "gold_answer": gold_index,
            "answer": pred,
            "count": 1,
            "source": source,
            "label": int(pred == gold_index) if pred is not None else 0,
            "raw_samples": raw,
            "raw_letters": letter,
        })
        row_idx += 1

    # Greedy decoding once
    idx, letter, raw = generate_answer_index(question, choices, do_sample=False)
    make_row(idx, "greedy", raw, letter)

    # Sampling NUM_SAMPLES times → store all separately
    for _ in range(NUM_SAMPLES):
        idx, letter, raw = generate_answer_index(question, choices, do_sample=True)
        make_row(idx, "sampling", raw, letter)

    # gold row
    make_row(
        gold_index,
        "gold",
        raw=None,
        letter=None
    )

    return rows

########################################
# Process dataset (tqdm: per question)
########################################

def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)

########################################
# Resume support
########################################

def load_tasks(input_path: str) -> List[Dict]:
    """Read the input file into (qid, question, choices, gold_index) tasks."""
    tasks = []

    with open(input_path, "r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            if args.realtimeQA:
                # questions without a gold answer are skipped entirely
                if "answer" not in ex:
                    continue
                tasks.append({
                    "qid": ex["question_id"],
                    "question": ex["question_sentence"],
                    "choices": ex["choices"],
                    "gold_index": int(ex["answer"][0]),
                })
            else:
                task = json.loads(ex["task_sample"])
                tasks.append({
                    "qid": f"{ex.get('entity', 'unknown')}_{line_idx}",
                    "question": task["question"],
                    "choices": task["options"],
                    "gold_index": ord(task["answer"].upper()) - ord("A"),
                })

    return tasks


def scan_completed(output_path: str) -> set:
    """
    Inspect an existing (possibly truncated) output file and return the set of
    qids that were fully written.

    A question is complete only once its "gold" row — the last row written for
    that question — is present. Rows of incomplete questions and any malformed
    trailing line are dropped, and the file is rewritten so that generation can
    safely append from there.
    """
    if not os.path.exists(output_path):
        return set()

    kept = []          # (qid, line) for every parsable row
    completed = set()
    broken_tail = False

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_id = row["id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                # only ever expected on the final line of a killed run
                broken_tail = True
                continue

            qid = row_id.rsplit("_", 1)[0]
            kept.append((qid, line))
            if row.get("source") == "gold":
                completed.add(qid)

    survivors = [line for qid, line in kept if qid in completed]
    dropped = len(kept) - len(survivors)

    if dropped or broken_tail:
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for line in survivors:
                f.write(line + "\n")
        os.replace(tmp_path, output_path)
        print(
            f"↩️  Trimmed {dropped} row(s) from an incomplete question "
            f"(+{'1 corrupt line' if broken_tail else 'no corrupt line'})"
        )

    return completed


def process_tasks(tasks: List[Dict], output_path: str, completed: set):
    """Generate candidates for the pending tasks, appending as we go."""
    pending = [t for t in tasks if t["qid"] not in completed]

    desc = "⚡ RealtimeQA questions" if args.realtimeQA else "📘 Processing questions"

    with tqdm(
        total=len(tasks),
        initial=len(tasks) - len(pending),
        desc=desc,
        file=sys.stderr,
        dynamic_ncols=True
    ) as qbar:

        with open(output_path, "a", encoding="utf-8") as fout:
            for t in pending:
                rows = build_candidate_rows(
                    t["qid"], t["question"], t["choices"], t["gold_index"]
                )
                for row in rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                # flush per question so a kill can lose at most one question
                fout.flush()
                os.fsync(fout.fileno())
                qbar.update(1)

########################################
# Run
########################################

if __name__ == "__main__":
    start = time.time()

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    tasks = load_tasks(INPUT_JSONL)

    if args.overwrite:
        if os.path.exists(OUTPUT_JSONL):
            os.remove(OUTPUT_JSONL)
        completed = set()
    else:
        completed = scan_completed(OUTPUT_JSONL)

    remaining = [t for t in tasks if t["qid"] not in completed]

    print(f"📂 Output: {OUTPUT_JSONL}")
    print(f"📊 {len(completed)}/{len(tasks)} questions already done, {len(remaining)} to go")

    if not remaining:
        print("✅ Already complete — skipping (use --overwrite to regenerate)")
        sys.exit(0)

    load_model()
    process_tasks(tasks, OUTPUT_JSONL, completed)

    elapsed = time.time() - start
    print(f"\n✅ Model: {args.model_size}")
    print(f"✅ Prompt ID: {args.prompt_id}")
    print(f"✅ Output: {OUTPUT_JSONL}")
    print(f"✅ Time: {elapsed/60:.2f} min")
