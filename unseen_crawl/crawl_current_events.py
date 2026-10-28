#!/usr/bin/env python3
"""
Crawl Wikipedia Portal:Current events day pages into an event-bullet corpus.

Two outputs, line-aligned 1:1:
  wiki_event_only.jsonl   {"text": ...}                     <- the CRAFT contract
  wiki_event_meta.jsonl   {"idx","date","section","revid","text"}

The main file stays text-only on purpose: create_task_samples_diff.py copies
every non-`text` key into meta, and the previous corpus was strictly
single-key, so extra fields would change downstream payloads.

Resumability is per-day HTML cache files, so an interrupted run resumes at the
first missing day and re-normalisation costs no requests.

Usage:
  python crawl_current_events.py --start 2025-03-26 --end 2026-07-29
  python crawl_current_events.py --no-network          # rebuild from cache
"""

import argparse
import datetime
import json
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import extract  # noqa: E402
import wiki_api  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "work", "00_corpus")
DEFAULT_CACHE = os.path.join(HERE, "cache")


def parse_date(s: str) -> datetime.date:
    if s == "today":
        return datetime.date.today()
    return datetime.date.fromisoformat(s)


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


def cache_path(cache_dir, d):
    return os.path.join(cache_dir, f"{d.isoformat()}.json")


def load_cache(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def save_cache(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def write_jsonl_atomic(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def crawl(dates, args):
    session = wiki_api.make_session(args.user_agent)
    tail_cutoff = args.end - datetime.timedelta(days=max(0, args.refresh_tail - 1))

    fetched = cached = missing = 0
    for d in tqdm(dates, desc="fetch", unit="day"):
        path = cache_path(args.cache_dir, d)
        stale_tail = d >= tail_cutoff
        if not args.force and not stale_tail and os.path.exists(path):
            if load_cache(path) is not None:
                cached += 1
                continue

        day = wiki_api.fetch_day(session, d, retries=args.retries,
                                 timeout=args.timeout)
        day["fetched_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        save_cache(path, day)
        if day.get("error"):
            missing += 1
        fetched += 1
        time.sleep(args.sleep)

    return {"fetched": fetched, "from_cache": cached, "missing_pages": missing}


def build_outputs(dates, args):
    seen = set()
    texts, metas = [], []
    per_day = {}
    totals = {}
    empty_days = []

    for d in dates:
        day = load_cache(cache_path(args.cache_dir, d))
        if day is None:
            empty_days.append({"date": d.isoformat(), "reason": "no_cache"})
            per_day[d.isoformat()] = 0
            continue

        rows, stats = extract.extract_day(
            day.get("html"), date=d.isoformat(),
            min_chars=args.min_chars, do_clean_spacing=args.clean_spacing,
            seen=seen,
        )
        for k, v in stats.items():
            if k == "reason":
                continue
            totals[k] = totals.get(k, 0) + v

        per_day[d.isoformat()] = len(rows)
        if not rows:
            empty_days.append({"date": d.isoformat(),
                               "reason": stats["reason"] or "no_bullets"})

        for r in rows:
            texts.append({"text": r["text"]})
            metas.append({
                "idx": len(metas),
                "date": r["date"],
                "section": r["section"],
                "revid": day.get("revid"),
                "text": r["text"],
            })

    os.makedirs(args.out_dir, exist_ok=True)
    main_path = os.path.join(args.out_dir, "wiki_event_only.jsonl")
    meta_path = os.path.join(args.out_dir, "wiki_event_meta.jsonl")
    write_jsonl_atomic(main_path, texts)
    write_jsonl_atomic(meta_path, metas)

    return {
        "bullets": len(texts),
        "days": len(per_day),
        "filters": totals,
        "empty_days": empty_days,
        "per_day": per_day,
        "outputs": {"main": main_path, "meta": meta_path},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-03-26")
    ap.add_argument("--end", default="2026-07-29")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--min-chars", type=int, default=extract.DEFAULT_MIN_CHARS)
    ap.add_argument("--refresh-tail", type=int, default=3,
                    help="always re-fetch the last N days (still being edited)")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--no-network", action="store_true",
                    help="rebuild outputs from cache only")
    ap.add_argument("--clean-spacing", action="store_true",
                    help="strip spaces before punctuation (changes ~81%% of lines)")
    ap.add_argument("--user-agent", default=wiki_api.DEFAULT_UA)
    ap.add_argument("--limit", type=int, default=0, help="first N days only")
    args = ap.parse_args()

    args.start = parse_date(args.start)
    args.end = parse_date(args.end)
    if args.end < args.start:
        ap.error("--end is before --start")

    dates = list(daterange(args.start, args.end))
    if args.limit:
        dates = dates[:args.limit]
        args.end = dates[-1]

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    started = time.time()
    fetch_stats = {"skipped": "no-network"}
    if not args.no_network:
        fetch_stats = crawl(dates, args)

    report = build_outputs(dates, args)
    report["fetch"] = fetch_stats
    report["range"] = [args.start.isoformat(), args.end.isoformat()]
    report["elapsed_sec"] = round(time.time() - started, 1)
    report["clean_spacing"] = args.clean_spacing

    report_path = os.path.join(args.out_dir, "crawl_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\ndays        : {report['days']}")
    print(f"bullets     : {report['bullets']}")
    print(f"filters     : {report['filters']}")
    print(f"empty days  : {len(report['empty_days'])}")
    print(f"elapsed     : {report['elapsed_sec']}s")
    print(f"-> {report['outputs']['main']}")
    print(f"-> {report_path}")


if __name__ == "__main__":
    main()
