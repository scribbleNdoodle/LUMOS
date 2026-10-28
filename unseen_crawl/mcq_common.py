"""
Shared MCQ helpers ported from CRAFT.

Sources (read-only originals, kept unmodified):
  CRAFT_new/code/utils/ts_creation.py            :64-114, :117-153, :543-584, :921-931
  CRAFT_new/code/create_task_samples_diff.py     :66-104, :363-370

Only the event-only path is ported. The group3 / csqa / recipegen /
summarization branches, the few-shot machinery (it was inert -- the diff
script always passes few_shots=None) and everything importing peft or thefuzz
are left behind.
"""

import json
import os
import re

import prompts


class QuestionLengthError(Exception):
    pass


class OptionsNumberError(Exception):
    pass


class AnswerFormatError(Exception):
    pass


# ---------------------------------------------------------------- io

def jsonl_iter(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def jsonl_write_atomic(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------- parsing

def qa_mc_extract(raw_text: str) -> dict:
    """Port of FormatExtractor.qa_mc, ts_creation.py:543-581.

    Scan every '{', raw_decode, keep dicts carrying question/options/answer,
    take the LAST one (handles "...rewrite it: { ... }" echoes).
    """
    decoder = json.JSONDecoder()
    candidates = []
    last_err = None

    for st, ch in enumerate(raw_text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw_text[st:])
        except Exception as e:  # noqa: BLE001 - mirrors the original
            last_err = e
            continue
        if (isinstance(obj, dict) and "question" in obj
                and "options" in obj and "answer" in obj):
            candidates.append(obj)

    if not candidates:
        raise AssertionError(f"No JSON found / parse failed. Last error: {last_err}")

    qa_dict = candidates[-1]

    question = str(qa_dict["question"]).strip()
    options = [str(opt).strip() for opt in qa_dict["options"]]
    answer = str(qa_dict["answer"])
    # The original indexes [0] unguarded; an empty answer is the model's
    # documented refusal token, so surface it as a normal format error.
    if not answer:
        raise AnswerFormatError("Empty answer (refusal output).")
    answer = answer[0].strip()

    if len(question) <= 15:
        raise QuestionLengthError("No question found / below 15 characters.")
    if len(options) != 4:
        raise OptionsNumberError("Number of answer options other than 4.")
    if answer not in ("A", "B", "C", "D"):
        raise AnswerFormatError("Answer does not include A, B, C, or D.")

    return {"question": question, "options": options, "answer": answer}


# ---------------------------------------------------------------- grounding

# Anchored, unlike create_task_samples_diff.py:66-70 which splits on the first
# ". " and therefore truncates any option whose own text contains ". ".
_OPT_RE = re.compile(r"^([A-D])\.\s*(.*)$", re.DOTALL)


def split_option(opt: str):
    """-> (letter, text) or (None, None) when the option is malformed."""
    m = _OPT_RE.match((opt or "").strip())
    if not m:
        return None, None
    return m.group(1), m.group(2).strip()


def answer_text_from_options(qa_dict: dict) -> str:
    """Port of create_task_samples_diff.py:77-91."""
    ans = (qa_dict.get("answer") or "").strip()
    if ans not in ("A", "B", "C", "D"):
        return ""
    for opt in qa_dict.get("options") or []:
        letter, text = split_option(opt)
        if letter == ans:
            return text
    return ""


def options_well_formed(qa_dict: dict) -> bool:
    letters = [split_option(o)[0] for o in qa_dict.get("options") or []]
    return letters == ["A", "B", "C", "D"]


def passes_answer_span_filter(qa_dict: dict, source_text: str,
                              relax: str = "none") -> bool:
    """Port of create_task_samples_diff.py:94-104.

    Default is the original case-sensitive substring test. `relax` only exists
    so filter_mcq.py can report what a looser rule would have kept.
    """
    if not source_text:
        return False
    atext = answer_text_from_options(qa_dict)
    if not atext:
        return False
    if relax == "none":
        return atext in source_text
    if relax == "case":
        return atext.lower() in source_text.lower()
    if relax == "ws":
        norm = lambda s: " ".join(s.lower().split())  # noqa: E731
        return norm(atext) in norm(source_text)
    raise ValueError(f"unknown relax mode: {relax}")


# ---------------------------------------------------------------- prompts

_PREAMBLE_TURN1 = (
    "You are an assistant that outputs ONLY valid JSON.\n"
    "Do not include any extra text.\n"
)

_PREAMBLE_TURN2 = (
    "You are an assistant that outputs ONLY valid JSON.\n"
    "Do not include any extra text.\n"
    "The output JSON must follow the same schema as the input DRAFT_JSON.\n\n"
)


def build_turn1_prompt(text: str) -> str:
    """ts_creation.py:64-114 with few_shots=None (the only path CRAFT used)."""
    return (
        _PREAMBLE_TURN1
        + "### Now you\n"
        + f"Instruction:\n{prompts.QA_MC_INSTRUCTION_EVENT_ONLY}\n\n"
          f"___________\n{text.strip()}\n\n"
        + "Output (JSON only):\n"
    )


def build_turn2_prompt(text: str, draft_json: str) -> str:
    """ts_creation.py:117-153."""
    payload = f"{text.strip()}\n\nDRAFT_JSON:\n{draft_json.strip()}\n"
    return (
        _PREAMBLE_TURN2
        + "### Instruction\n"
        + f"{prompts.QA_MC_REVIEW_INSTRUCTION_EVENT_ONLY}\n\n"
        + "___________\n"
        + payload
        + "\n"
        + "### Output (JSON only)\n"
    )


def to_messages(prompt_body: str, preamble: str):
    """Split a CRAFT plain-text prompt into chat messages.

    CRAFT fed these strings to LLM.generate with no chat template, so an
    -Instruct model had to *continue* text rather than answer it. Routing the
    preamble to `system` and the rest to `user` puts the model back in its
    trained regime.
    """
    body = prompt_body[len(preamble):] if prompt_body.startswith(preamble) else prompt_body
    return [
        {"role": "system", "content": preamble.strip()},
        {"role": "user", "content": body},
    ]


def turn1_messages(text: str):
    return to_messages(build_turn1_prompt(text), _PREAMBLE_TURN1)


def turn2_messages(text: str, draft_json: str):
    return to_messages(build_turn2_prompt(text, draft_json), _PREAMBLE_TURN2)


# ---------------------------------------------------------------- record

def clean_record(qa_dict: dict, meta: dict) -> dict:
    """The downstream contract (create_task_samples_diff.py:363-370).

    `task_sample` is a JSON *string*; cA_mcq.py:487 re-parses it.
    """
    return {
        "task_sample": json.dumps(qa_dict, ensure_ascii=False),
        "entity": None,
        "chunks": None,
        "meta": meta,
    }
