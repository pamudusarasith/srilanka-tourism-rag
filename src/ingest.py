"""
Build the knowledge base: web sources into PostgreSQL.

Structured fields come from data/seeds/attractions.csv, prose and photographs
from the web. Each run rebuilds the database and commits per attraction, so a
network failure keeps the earlier work.

Run:  uv run python src/ingest.py
"""

import csv
import sys

import psycopg

import config
import sources


def chunk_text(text: str, name: str) -> list[str]:
    """
    Split an article into overlapping word windows.

    Chunks are prefixed with the attraction name. Without it, a chunk reading
    "It is 263 m high" says nothing once retrieved on its own.
    """
    words = text.split()
    chunks, start = [], 0
    step = config.CHUNK_WORDS - config.CHUNK_OVERLAP_WORDS
    while start < len(words):
        window = words[start : start + config.CHUNK_WORDS]
        if len(window) < 30 and chunks:
            break
        chunks.append(f"{name}: " + " ".join(window))
        start += step
    return chunks


def to_num(v):
    return float(v) if v not in (None, "") else None


def to_int(v):
    return int(v) if v not in (None, "") else None


def to_bool(v):
    return str(v).strip().lower() == "true"


def blank_to_none(v):
    v = (v or "").strip()
    return v or None


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


def insert_attraction(cur, row: dict, wiki: dict) -> int:
    cur.execute(
        """
        INSERT INTO attractions
            (name, category, district, province, latitude, longitude,
             height_m, trekking_difficulty, swimming_safety, surf_season,
             era, unesco_status, dress_code, entrance_fee_lkr,
             best_season, accessibility, summary, wiki_url)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """,
        (
            row["name"],
            row["category"],
            blank_to_none(row["district"]),
            blank_to_none(row["province"]),
            wiki["lat"],
            wiki["lon"],
            to_num(row["height_m"]),
            blank_to_none(row["trekking_difficulty"]),
            blank_to_none(row["swimming_safety"]),
            blank_to_none(row["surf_season"]),
            blank_to_none(row["era"]),
            to_bool(row["unesco_status"]),
            blank_to_none(row["dress_code"]),
            to_int(row["entrance_fee_lkr"]),
            blank_to_none(row["best_season"]),
            blank_to_none(row["accessibility"]),
            wiki["text"].split("\n")[0][:1500],
            wiki["url"],
        ),
    )
    return cur.fetchone()[0]


def insert_chunks(
    cur,
    attraction_id: int,
    name: str,
    text: str,
    source: str,
    url: str,
    start_index: int,
) -> int:
    chunks = chunk_text(text, name)
    for offset, chunk in enumerate(chunks):
        cur.execute(
            """INSERT INTO documents
                   (attraction_id, chunk_index, chunk_text, source, source_url)
               VALUES (%s,%s,%s,%s,%s)""",
            (attraction_id, start_index + offset, chunk, source, url),
        )
    return len(chunks)


def insert_images(cur, attraction_id: int, row: dict) -> int:
    candidates = []
    try:
        candidates = sources.search_commons_images(row["name"])
    except Exception as exc:  # noqa: BLE001
        print(f"      ! Commons search failed ({str(exc)[:60]})")

    saved = 0
    for meta in candidates:
        if saved >= config.IMAGES_PER_ATTRACTION:
            break
        dest = config.IMAGES_DIR / slugify(row["name"]) / f"{saved:02d}.jpg"
        if not sources.download_image(meta["url"], dest):
            continue

        # Held out of the vector index to form the image-retrieval eval set.
        # Restricting it to waterfalls keeps the metric meaningful: CLIP tells
        # a beach from a temple trivially, two waterfalls much less so.
        held_out = row["category"] == "waterfall" and saved == 3

        cur.execute(
            """INSERT INTO images
                   (attraction_id, file_path, caption, source_url, license, is_held_out)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (
                attraction_id,
                str(dest.relative_to(config.ROOT)),
                meta["title"].replace("File:", "").rsplit(".", 1)[0].replace("_", " "),
                meta["descriptionurl"],
                f'{meta["license"]} | {meta["artist"]}'.strip(" |"),
                held_out,
            ),
        )
        saved += 1
    return saved


def main() -> None:
    seeds_path = config.SEEDS_DIR / "attractions.csv"
    with open(seeds_path, newline="", encoding="utf-8") as fh:
        seeds = list(csv.DictReader(fh))
    print(f"Loaded {len(seeds)} seed rows from {seeds_path.name}\n")

    conn = psycopg.connect(config.DB_CONN)
    conn.autocommit = False

    with conn.cursor() as cur:
        cur.execute((config.ROOT / "src" / "schema.sql").read_text())
    conn.commit()
    print("Schema applied.\n")

    n_attr = n_img = n_doc = n_al = 0
    failures = []

    for row in seeds:
        name = row["name"]
        print(f"[{name}]", flush=True)

        try:
            wiki = sources.fetch_wikipedia(row["wiki_title"])
        except Exception as exc:  # noqa: BLE001
            print(f"      ! Wikipedia fetch failed ({str(exc)[:70]}) -- skipped")
            conn.rollback()
            failures.append(name)
            continue

        if wiki is None:
            print("      ! no Wikipedia page found -- skipped")
            failures.append(name)
            continue

        try:
            with conn.cursor() as cur:
                attraction_id = insert_attraction(cur, row, wiki)

                chunk_i = insert_chunks(
                    cur,
                    attraction_id,
                    name,
                    wiki["text"],
                    "wikipedia",
                    wiki["url"],
                    0,
                )
                print(
                    f"      {chunk_i} chunks (wikipedia, {len(wiki['text'].split())} words)"
                )

                amazing = sources.fetch_amazinglanka(name)
                if amazing:
                    added = insert_chunks(
                        cur,
                        attraction_id,
                        name,
                        amazing["text"],
                        "amazinglanka",
                        amazing["url"],
                        chunk_i,
                    )
                    chunk_i += added
                    n_al += 1
                    print(
                        f"      {added} chunks (amazinglanka, "
                        f"{len(amazing['text'].split())} words)"
                    )

                saved = insert_images(cur, attraction_id, row)
                print(f"      {saved} images", flush=True)

            conn.commit()
            n_attr += 1
            n_doc += chunk_i
            n_img += saved
        except Exception as exc:  # noqa: BLE001
            print(f"      ! failed ({str(exc)[:70]}) -- rolled back")
            conn.rollback()
            failures.append(name)

    print("\n" + "=" * 62)
    print(f"attractions            : {n_attr}")
    print(f"images                 : {n_img}")
    print(f"document chunks        : {n_doc}")
    print(f"with AmazingLanka text : {n_al}/{n_attr}")
    if failures:
        print(f"FAILED                 : {', '.join(failures)}")
    print("=" * 62)

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
