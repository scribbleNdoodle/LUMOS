"""
Wikipedia API layer for the Portal:Current events crawl.

Network only — no parsing lives here (see extract.py).

The day-page title format is `Portal:Current events/{YYYY} {Month} {D}`
(no zero padding), e.g. `Portal:Current events/2025 March 26`.
Month names are hardcoded on purpose: `strftime("%B")` is locale-dependent.
"""

import datetime
import random
import time

import requests

API = "https://en.wikipedia.org/w/api.php"

DEFAULT_UA = (
    "lumos-unseen-crawl/1.0 (research; pathssun@gmail.com) python-requests/2.32"
)

MONTHS = (
    None,
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# API errors that mean "this page will never exist" — do not retry.
TERMINAL_ERRORS = {"missingtitle", "invalidtitle", "nosuchpageid"}
RETRY_STATUS = {429, 500, 502, 503, 504}


def day_title(d: datetime.date) -> str:
    return f"Portal:Current events/{d.year} {MONTHS[d.month]} {d.day}"


def make_session(user_agent: str = DEFAULT_UA) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip"})
    return s


def _backoff(attempt: int) -> float:
    return min(60.0, 2.0 ** attempt) + random.uniform(0, 0.5)


def fetch_parse(session, title, retries=5, timeout=30):
    """Return the parse payload for `title`, or None if the page does not exist.

    Raises RuntimeError once the retry budget is exhausted on a transient error.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "text|revid",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "disablelimitreport": "1",
        "disableeditsection": "1",
        "disabletoc": "1",
        "maxlag": "5",
    }

    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=timeout)
        except requests.RequestException as e:
            last_err = f"request failed: {e}"
            time.sleep(_backoff(attempt))
            continue

        if r.status_code in RETRY_STATUS:
            last_err = f"HTTP {r.status_code}"
            wait = r.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else _backoff(attempt))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"{title}: HTTP {r.status_code}")

        try:
            j = r.json()
        except ValueError as e:
            last_err = f"bad json: {e}"
            time.sleep(_backoff(attempt))
            continue

        err = j.get("error")
        if err:
            code = err.get("code", "")
            if code in TERMINAL_ERRORS:
                return None
            last_err = f"api error: {code}"
            time.sleep(_backoff(attempt))
            continue

        return j.get("parse")

    raise RuntimeError(f"{title}: giving up after {retries} attempts ({last_err})")


def fetch_day(session, d: datetime.date, retries=5, timeout=30) -> dict:
    title = day_title(d)
    parse = fetch_parse(session, title, retries=retries, timeout=timeout)
    if parse is None:
        return {"date": d.isoformat(), "title": title, "revid": None,
                "html": None, "error": "missingtitle"}
    return {
        "date": d.isoformat(),
        "title": title,
        "revid": parse.get("revid"),
        "html": parse.get("text") or "",
        "error": None,
    }


def existing_titles(session, dates, timeout=30) -> set:
    """Pre-flight existence check, 50 titles per request."""
    found = set()
    titles = [day_title(d) for d in dates]
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        params = {
            "action": "query",
            "titles": "|".join(chunk),
            "format": "json",
            "formatversion": "2",
        }
        r = session.get(API, params=params, timeout=timeout)
        r.raise_for_status()
        for page in r.json().get("query", {}).get("pages", []):
            if not page.get("missing"):
                found.add(page.get("title"))
        time.sleep(0.2)
    return found
