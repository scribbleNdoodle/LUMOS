"""
Portal:Current events HTML -> event bullets.

Pure functions, no network: the crawler caches raw HTML per day, so
re-normalising the whole corpus costs zero requests.

The normalisation below was reverse-engineered from the previous corpus
(assets/unseen_web/corpus_samples/wiki_event_only.jsonl) and reproduces it
byte-for-byte:

  * take only *leaf* <li> inside div.current-events-content
  * text = " ".join(li.get_text(separator=" ").split())

`separator=" "` is what produces the corpus' characteristic spacing
("led by AFRICOM .", "( The Guardian )", "Cardel- Nautla ,") — MediaWiki
renders link trails inside the <a>, so adjacent text nodes get joined with a
space. Dropping the separator would silently produce a differently-styled
corpus, so verify_crawl.py checks for it explicitly.
"""

import re

from bs4 import BeautifulSoup

NAV_CLASSES = (
    "current-events-navbar", "current-events-nav", "current-events-more",
    "editlink", "noprint", "navbox", "reflist", "mw-editsection", "toc",
)

NAV_LITERALS = {
    "edit", "history", "watch", "purge", "see also", "more current events...",
}

_HAS_ALPHA = re.compile(r"[A-Za-z]")

DEFAULT_MIN_CHARS = 40
DEFAULT_MIN_WORDS = 5


def normalize_text(node) -> str:
    return " ".join(node.get_text(separator=" ").split())


def clean_spacing(text: str) -> str:
    """Optional cosmetic pass. Off by default -- it rewrites ~81% of lines."""
    text = re.sub(r"\s+([.,;:!?%])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def extract_day(html, date=None, min_chars=DEFAULT_MIN_CHARS,
                min_words=DEFAULT_MIN_WORDS, do_clean_spacing=False, seen=None):
    """Return (rows, stats).

    rows: [{"date", "section", "text"}] in document order.
    stats: filter counters + a `reason` key when nothing could be extracted.
    """
    stats = {"empty": 0, "too_short": 0, "no_alpha": 0, "too_few_words": 0,
             "nav_literal": 0, "dup": 0, "kept": 0, "reason": None}
    rows = []

    if not html:
        stats["reason"] = "no_html"
        return rows, stats

    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.current-events-content")
    if content is None:
        stats["reason"] = "no_content_div"
        return rows, stats

    # Defensive: measured to be a no-op today, cheap insurance against
    # template changes that would pull nav links into the content div.
    selector = ", ".join("." + c for c in NAV_CLASSES)
    for bad in content.select(selector):
        bad.decompose()

    current_section = None
    for child in content.find_all(recursive=False):
        if child.name == "p":
            s = normalize_text(child)
            if s:
                current_section = s
            continue
        if child.name not in ("ul", "ol"):
            continue

        for li in child.find_all("li"):
            # Non-leaf <li> are topic-navigation wrappers, never sourced events.
            if li.find("li"):
                continue

            text = normalize_text(li)
            if do_clean_spacing:
                text = clean_spacing(text)

            if not text:
                stats["empty"] += 1
                continue
            if text.lower() in NAV_LITERALS:
                stats["nav_literal"] += 1
                continue
            if len(text) < min_chars:
                stats["too_short"] += 1
                continue
            if not _HAS_ALPHA.search(text):
                stats["no_alpha"] += 1
                continue
            if len(text.split()) < min_words:
                stats["too_few_words"] += 1
                continue
            if seen is not None:
                if text in seen:
                    stats["dup"] += 1
                    continue
                seen.add(text)

            rows.append({"date": date, "section": current_section, "text": text})
            stats["kept"] += 1

    return rows, stats
