#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VENDORED COPY -- do not edit except where marked.

source : /mnt/PK/dataset/realtimeqa_public/2.judge_realtime.py
sha256 : 35eaab496da356b12627ffe4074be50747aa499c71954b24e026032a868c3095
copied : 2026-07-30

Vendored because /mnt/PK/dataset is a world-writable share; the original can be
edited or moved out from under us.

Only change vs. the original: --backend defaults to "none" (heuristic only)
because OPENAI_API_KEY is not set on this machine. Measured impact on the
previous unseen_wiki data -- the LLM pass changed 11/6069 (7B), 15/5459 (13B),
19/5238 (32B) candidate labels, flipping 6/8/8 questions from miss to hit,
because 94% of golds are short enough that is_short_gold() skips the LLM
entirely. Re-runnable with --backend openai if a key appears.
"""

import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

try:
    from openai import AsyncOpenAI  # pip install openai
except Exception:
    AsyncOpenAI = None


# -----------------------
# 1) Normalization / Heuristics
# -----------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\s\"'“”‘’\(\)\[\]\{\}]+|[\s\"'“”‘’\(\)\[\]\{\}]+$")
_TRAILING_DOT_RE = re.compile(r"[\.。]+$")


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_answer(s: str) -> str:
    """
    Conservative normalization for entity/phrase matching:
    - lowercase
    - strip accents
    - trim edge quotes/brackets
    - collapse whitespace
    - remove trailing dots
    """
    if s is None:
        return ""
    x = str(s).strip()
    if not x:
        return ""
    x = x.replace("’", "'").replace("`", "'")
    x = _strip_accents(x).lower()
    x = _PUNCT_EDGE_RE.sub("", x)
    x = _WS_RE.sub(" ", x).strip()
    x = _TRAILING_DOT_RE.sub("", x).strip()
    return x


def normalize_possessive(x: str) -> str:
    if not x:
        return x
    if x.endswith("'s"):
        return x[:-2].strip()
    if x.endswith("s'"):
        return (x[:-2].strip() + "s").strip()
    return x


def strip_article_if_short(z: str) -> str:
    for art in ("the ", "a ", "an "):
        if z.startswith(art):
            return z[len(art):].strip()
    return z


def heuristic_match(gold: str, cand: str) -> bool:
    g = normalize_possessive(normalize_answer(gold))
    c = normalize_possessive(normalize_answer(cand))
    if not g or not c:
        return False
    if g == c:
        return True

    # conservative article stripping for short entities only
    if len(g.split()) <= 4 and len(c.split()) <= 4:
        if strip_article_if_short(g) == strip_article_if_short(c):
            return True

    return False


def token_set(s: str) -> set:
    # Very simple tokenization (works fine for your short answers)
    s = normalize_possessive(normalize_answer(s))
    if not s:
        return set()
    return set(s.split())


def is_short_gold(gold: str, max_tokens: int = 4, max_chars: int = 40) -> bool:
    g = normalize_possessive(normalize_answer(gold))
    if not g:
        return True
    return (len(g.split()) <= max_tokens) and (len(g) <= max_chars)


# -----------------------
# 2) Dedupe candidates inside a record + broadcast
# -----------------------

def dedupe_candidates(cands: List[dict]) -> Tuple[List[str], Dict[int, List[int]]]:
    """
    Returns:
      uniq_texts: unique candidate texts (first occurrence raw text)
      map_uniq_to_indices: uniq_idx -> list of original candidate indices
    Dedup key is normalized candidate.
    """
    uniq_texts: List[str] = []
    map_uniq_to_indices: Dict[int, List[int]] = {}
    seen: Dict[str, int] = {}

    for i, c in enumerate(cands):
        ans = str(c.get("answer", "") or "")
        norm = normalize_possessive(normalize_answer(ans))
        if not norm:
            continue
        if norm in seen:
            ui = seen[norm]
        else:
            ui = len(uniq_texts)
            seen[norm] = ui
            uniq_texts.append(ans)
        map_uniq_to_indices.setdefault(ui, []).append(i)

    return uniq_texts, map_uniq_to_indices


# -----------------------
# 3) LLM judge for LONG gold (quote guardrail)
# -----------------------

def build_prompt_quote_guardrail(question: str, gold: str, candidates: List[str]) -> str:
    items = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
    return (
        "You are an answer-matching judge evaluating semantic equivalence.\n"
        "For each candidate, output label=1 if it conveys the SAME MEANING or factual information as the gold answer.\n"
        "Allow for paraphrasing, synonyms, different sentence structures, or extra details as long as the core answer is correct.\n"
        "If label=1, you MUST provide a supporting quote copied verbatim from the candidate text that contains the answer.\n"
        "If the candidate contradicts the gold answer or misses the key information, label MUST be 0.\n\n"
        f"Q: {question}\n"
        f"GOLD: {gold}\n"
        "CANDIDATES:\n"
        f"{items}\n\n"
        "Return STRICT JSON only (no extra keys):\n"
        "{\"results\":[{\"idx\":1,\"label\":0,\"quote\":\"\"}]}"
    )


def parse_results_json(text: str, n: int) -> List[Tuple[int, str]]:
    """
    Returns list of (label, quote) length n. Defaults label=0, quote="".
    """
    out = [(0, "") for _ in range(n)]
    if not text:
        return out

    t = text.strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        t = m.group(0)

    try:
        obj = json.loads(t)
        results = obj.get("results", [])
        if isinstance(results, list):
            for item in results:
                try:
                    idx = int(item.get("idx"))
                    if not (1 <= idx <= n):
                        continue
                    lab_raw = item.get("label", 0)
                    lab = 1 if (lab_raw is True or str(lab_raw).strip() == "1") else 0
                    quote = str(item.get("quote", "") or "")
                    out[idx - 1] = (lab, quote)
                except Exception:
                    continue
    except Exception:
        pass

    return out


def postcheck_label_with_quote(
    gold: str,
    cand: str,
    llm_label: int,
    quote: str,
    min_token_overlap: int = 1,
) -> int:
    """
    Hard guardrails to prevent 'Elephant' getting label=1 for gold='Mouse':
    - if llm_label==1, quote must be non-empty and must be substring of candidate
    - quote must have some token overlap with gold (conservative)
    """
    if llm_label != 1:
        return 0

    if not quote.strip():
        return 0

    cand_raw = str(cand or "")
    if quote not in cand_raw:
        # must be verbatim span copied from candidate
        return 0

    gset = token_set(gold)
    qset = token_set(quote)
    if not gset or not qset:
        return 0

    overlap = len(gset & qset)
    if overlap < min_token_overlap:
        return 0

    return 1


async def openai_call_with_retry(fn, retries: int = 4, base_sleep: float = 0.6):
    for t in range(retries + 1):
        try:
            return await fn()
        except Exception:
            if t == retries:
                raise
            await asyncio.sleep(base_sleep * (2 ** t))


async def judge_long_gold_openai(
    client: Any,
    model: str,
    question: str,
    gold: str,
    cand_texts: List[str],
    semaphore: asyncio.Semaphore,
    max_tokens: int = 300,
) -> List[Tuple[int, str]]:
    prompt = build_prompt_quote_guardrail(question, gold, cand_texts)

    async with semaphore:
        async def _call():
            return await client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
            )

        resp = await openai_call_with_retry(_call)

    txt = resp.choices[0].message.content or ""
    return parse_results_json(txt, len(cand_texts))


# -----------------------
# 4) Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL (one question per line)")
    parser.add_argument("--output", required=True, help="Output JSONL (same, labels injected into candidates)")
    # CHANGED from "openai": no OPENAI_API_KEY on this machine. See module docstring.
    parser.add_argument("--backend", choices=["openai", "none"], default="none")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--batch_candidates", type=int, default=25, help="LLM chunk size per question (unique candidates)")
    parser.add_argument("--short_gold_max_tokens", type=int, default=4)
    parser.add_argument("--short_gold_max_chars", type=int, default=40)
    args = parser.parse_args()

    if args.backend == "openai" and AsyncOpenAI is None:
        print("Error: openai not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    in_path = Path(args.input)
    out_path = Path(args.output)

    records: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # cache: (gold_norm, cand_norm) -> label
    cache: Dict[Tuple[str, str], int] = {}

    # LLM tasks: (record_index, uniq_indices, uniq_texts)
    llm_tasks: List[Tuple[int, List[int], List[str]]] = []

    # 1) Initialize labels + heuristic + decide LLM tasks
    for r_i, rec in enumerate(tqdm(records, desc="Pass1: heuristic + task build")):
        q = rec.get("question", "") or ""
        gold = rec.get("gold_answer", "") or ""
        cands = rec.get("candidates", []) or []

        # ensure label field exists
        for c in cands:
            c["label"] = 0

        if not str(gold).strip():
            continue

        short_gold = is_short_gold(
            gold,
            max_tokens=args.short_gold_max_tokens,
            max_chars=args.short_gold_max_chars,
        )

        gold_norm = normalize_possessive(normalize_answer(gold))

        # dedupe candidates
        uniq_texts, map_uniq_to_orig = dedupe_candidates(cands)

        # For each unique cand, decide label or mark ambiguous
        uniq_labels: List[Optional[int]] = [None] * len(uniq_texts)

        for ui, ans_text in enumerate(uniq_texts):
            if not str(ans_text).strip():
                uniq_labels[ui] = 0
                continue

            # heuristic exact-ish match
            if heuristic_match(gold, ans_text):
                uniq_labels[ui] = 1
                continue

            cand_norm = normalize_possessive(normalize_answer(ans_text))
            if not cand_norm:
                uniq_labels[ui] = 0
                continue

            ck = (gold_norm, cand_norm)
            if ck in cache:
                uniq_labels[ui] = cache[ck]
                continue

            # IMPORTANT:
            # - If gold is short, DO NOT use LLM. Everything non-match remains 0.
            if short_gold:
                uniq_labels[ui] = 0
            else:
                # long gold: allow LLM
                uniq_labels[ui] = None

        # broadcast decided labels back to original candidates
        for ui, lab in enumerate(uniq_labels):
            if lab is None:
                continue
            for orig_i in map_uniq_to_orig.get(ui, []):
                cands[orig_i]["label"] = int(lab)

            cand_norm = normalize_possessive(normalize_answer(uniq_texts[ui]))
            if gold_norm and cand_norm:
                cache[(gold_norm, cand_norm)] = int(lab)

        # build LLM tasks for remaining None (long gold only)
        if args.backend == "openai" and (not short_gold):
            ambiguous_uis = [ui for ui, lab in enumerate(uniq_labels) if lab is None]
            if ambiguous_uis:
                # chunk by args.batch_candidates
                for start in range(0, len(ambiguous_uis), args.batch_candidates):
                    chunk_uis = ambiguous_uis[start:start + args.batch_candidates]
                    chunk_texts = [uniq_texts[ui] for ui in chunk_uis]
                    llm_tasks.append((r_i, chunk_uis, chunk_texts))

    # 2) LLM pass (long gold only) with quote guardrail + postcheck
    if args.backend == "openai" and llm_tasks:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        async def process_one(task):
            r_i, chunk_uis, chunk_texts = task
            rec = records[r_i]
            q = rec.get("question", "") or ""
            gold = rec.get("gold_answer", "") or ""
            cands = rec.get("candidates", []) or []

            # need dedupe mapping again to broadcast (cheap)
            uniq_texts, map_uniq_to_orig = dedupe_candidates(cands)
            # align: chunk_uis indexes refer to uniq_texts from pass1; should still match by normalization
            # (because we recompute in same way) so safe.

            results = await judge_long_gold_openai(
                client=client,
                model=args.model,
                question=q,
                gold=gold,
                cand_texts=chunk_texts,
                semaphore=sem,
            )

            gold_norm = normalize_possessive(normalize_answer(gold))

            for local_idx, ui in enumerate(chunk_uis):
                ans_text = chunk_texts[local_idx]
                llm_lab, quote = results[local_idx] if local_idx < len(results) else (0, "")

                # postcheck hard guardrail
                final_lab = postcheck_label_with_quote(
                    gold=gold,
                    cand=ans_text,
                    llm_label=llm_lab,
                    quote=quote,
                    min_token_overlap=1,
                )

                # broadcast to original candidates
                for orig_i in map_uniq_to_orig.get(ui, []):
                    rec["candidates"][orig_i]["label"] = int(final_lab)

                cand_norm = normalize_possessive(normalize_answer(ans_text))
                if gold_norm and cand_norm:
                    cache[(gold_norm, cand_norm)] = int(final_lab)

        async def run_all():
            # parallelize tasks with gather (bounded by semaphore inside call)
            # chunking helps prevent huge prompts; concurrency controls throughput.
            coros = [process_one(t) for t in llm_tasks]
            for fut in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Pass2: OpenAI judging"):
                await fut

        asyncio.run(run_all())

    # 3) Save as question-per-line JSONL (labels injected)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(records)} questions to {out_path} (labels injected into candidates).")


if __name__ == "__main__":
    main()