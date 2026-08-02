"""
Wikipedia, AmazingLanka and Wikimedia Commons clients.

Wikipedia's articles on many Sri Lankan waterfalls are stubs, so AmazingLanka
supplies the missing detail. Images come from Commons only, since
AmazingLanka's photographs are copyrighted.
"""

import html
import io
import re
import time
from difflib import SequenceMatcher

import requests
from PIL import Image

import config

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": config.USER_AGENT})

SKIP_IMAGE_HINTS = (
    "icon",
    "logo",
    "map",
    "flag",
    "commons-",
    "wiki",
    "symbol",
    "edit-",
    "ambox",
    "question_book",
    "location",
    "coat_of_arms",
    "disambig",
    "sound",
    "speaker",
)


def api_get(url: str, params: dict | None = None, attempts: int = 5) -> dict | list:
    """GET a JSON API, backing off on 429 and retrying network errors."""
    delay = 2.0
    for _ in range(attempts):
        try:
            r = SESSION.get(url, params=params, timeout=45)
        except (requests.Timeout, requests.ConnectionError) as exc:
            print(
                f"      . network error ({type(exc).__name__}), retry in {delay:.0f}s"
            )
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code in (429, 503):
            wait = float(r.headers.get("Retry-After", delay))
            print(f"      . rate-limited, waiting {wait:.0f}s")
            time.sleep(wait)
            delay *= 2
            continue
        r.raise_for_status()
        time.sleep(config.API_DELAY)
        return r.json()
    raise RuntimeError(f"giving up on {url} after {attempts} attempts")


def strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style|figure|figcaption)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p>", "\n\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def fetch_wikipedia(title: str) -> dict | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|coordinates|info",
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "titles": title,
        "redirects": 1,
    }
    pages = api_get(config.WIKI_API, params).get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid == "-1" or "extract" not in page:
            return None
        coords = (page.get("coordinates") or [{}])[0]
        return {
            "title": page.get("title", title),
            "text": _trim_tail(page["extract"]),
            "lat": coords.get("lat"),
            "lon": coords.get("lon"),
            "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}"),
        }
    return None


def _trim_tail(text: str) -> str:
    for marker in (
        "\nSee also",
        "\nReferences",
        "\nExternal links",
        "\nNotes",
        "\nFurther reading",
        "\nGallery",
    ):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def fetch_amazinglanka(name: str, min_similarity: float = 0.45) -> dict | None:
    """
    Fetch an attraction's AmazingLanka article through its WordPress REST API.

    WordPress search is fuzzy. Querying "Bambarakanda" returns the Devil's
    Staircase trail high in the results, and attaching the wrong article is
    worse than attaching none, hence the title-similarity threshold.
    """
    try:
        posts = api_get(
            config.AMAZINGLANKA_API,
            {"search": name, "per_page": 5, "_fields": "title,content,link,slug"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      . AmazingLanka lookup failed ({str(exc)[:60]})")
        return None

    if not isinstance(posts, list) or not posts:
        return None

    target = _normalise(name)
    best, best_score = None, 0.0
    for post in posts:
        title = strip_html(post.get("title", {}).get("rendered", ""))
        score = SequenceMatcher(None, target, _normalise(title)).ratio()
        if target in _normalise(title):
            score = max(score, 0.9)
        if score > best_score:
            best, best_score = post, score

    if best is None or best_score < min_similarity:
        return None

    text = _clean_amazinglanka(strip_html(best.get("content", {}).get("rendered", "")))
    if len(text.split()) < 60:
        return None

    return {
        "title": strip_html(best.get("title", {}).get("rendered", "")),
        "text": text,
        "url": best.get("link", ""),
    }


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _clean_amazinglanka(text: str) -> str:
    # The page's star-rating widget renders as text and would otherwise be
    # embedded along with the article.
    text = re.sub(
        r"RATE THIS LOCATION\s*:.*?Loading\.\.\.\s*", "", text, flags=re.S | re.I
    )
    for marker in (
        "Also See",
        "Please help us",
        "Share this",
        "Related posts",
        "Leave a Reply",
        "You may also like",
    ):
        idx = text.find(marker)
        if idx > 200:
            text = text[:idx]
    return text.strip()


def search_commons_images(query: str, limit: int = 14) -> list[dict]:
    """
    Search Commons for photographs.

    The article's own `prop=images` lists only the few photos embedded in it,
    plus icons and locator maps.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} Sri Lanka",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": config.IMAGE_THUMB_WIDTH,
    }
    pages = api_get(config.COMMONS_API, params).get("query", {}).get("pages", {})

    results = []
    for page in pages.values():
        title = page.get("title", "")
        low = title.lower()
        if not low.endswith((".jpg", ".jpeg", ".png")):
            continue
        if any(h in low for h in SKIP_IMAGE_HINTS):
            continue
        ii = (page.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        if not url:
            continue
        meta = ii.get("extmetadata", {})
        results.append(
            {
                "title": title,
                "url": url,
                "descriptionurl": ii.get("descriptionurl", ""),
                "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
                "artist": strip_html(meta.get("Artist", {}).get("value", "")),
                "index": page.get("index", 999),
            }
        )
    results.sort(key=lambda d: d["index"])
    return results


def download_image(url: str, dest_path) -> bool:
    """Download and resize one photograph."""
    for attempt in range(config.IMAGE_RETRIES):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 5 * (attempt + 1)))
                print(f"      . CDN throttled, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.thumbnail((config.IMAGE_MAX_PX, config.IMAGE_MAX_PX))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest_path, "JPEG", quality=88)
            time.sleep(config.IMAGE_DELAY)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"      ! image failed ({str(exc)[:70]})")
            return False
    return False
