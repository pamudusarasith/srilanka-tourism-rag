"""
Router accuracy, text retrieval, image retrieval, and an ablation of SQL-only
against semantic-only against the full pipeline.

Writes CSVs to report/results/.

Run:  uv run python src/evaluate.py
"""

import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg
from PIL import Image

import config
import generate
import pipeline
import retrieval

RESULTS_DIR = config.ROOT / "report" / "results"

# Router and ablation questions are independent of each other, so they run
# concurrently. Measured empirically: 4 concurrent workers was slower than 1
# (free-tier rate limiting silently eats the parallelism gain in retry
# backoff), 2 was the fastest of the three tried.
EVAL_CONCURRENCY = 2


def load_questions() -> list[dict]:
    path = config.SEEDS_DIR / "eval_questions.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["expected"] = [
            n.strip() for n in r["expected_attractions"].split("|") if n.strip()
        ]
    return rows


def name_to_id() -> dict[str, int]:
    with psycopg.connect(config.DB_CONN) as conn, conn.cursor() as cur:
        cur.execute("SELECT name, id FROM attractions")
        return {name: aid for name, aid in cur.fetchall()}


def write_csv(filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path.relative_to(config.ROOT)}")


def first_correct_rank(ranked_ids: list[int], expected_ids: set[int]) -> int | None:
    for i, aid in enumerate(ranked_ids, start=1):
        if aid in expected_ids:
            return i
    return None


def mean_of_scored(rows: list[dict], key: str) -> float:
    """Mean over scored rows. Blanks are runs excluded by an API failure."""
    scored = [r[key] for r in rows if r.get(key) != ""]
    return sum(scored) / len(scored) if scored else 0.0


def _classify(q: dict) -> str:
    return retrieval.route(q["question"], has_image=False)["query_type"]


def eval_router(questions: list[dict]) -> list[dict]:
    print("\n[1] Router accuracy")
    rows, correct = [], 0
    confusion = defaultdict(int)

    # as_completed rather than pool.map(), so each result prints as soon as
    # it lands instead of only after every question has finished -- with 36
    # concurrent questions a run without this looks identical to a hang.
    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as pool:
        future_to_q = {pool.submit(_classify, q): q for q in questions}
        by_id = {}
        for future in as_completed(future_to_q):
            q = future_to_q[future]
            predicted = future.result()
            ok = predicted == q["expected_type"]
            correct += ok
            confusion[(q["expected_type"], predicted)] += 1
            row = {
                "id": q["id"],
                "question": q["question"],
                "expected": q["expected_type"],
                "predicted": predicted,
                "correct": int(ok),
            }
            by_id[q["id"]] = row
            print(
                f"  {q['id']}  {q['expected_type']:>10} -> {predicted:<10} "
                f"{'ok' if ok else 'MISS'}"
            )

    rows = [by_id[q["id"]] for q in questions]
    print(f"  accuracy: {correct}/{len(questions)} = {correct / len(questions):.1%}")
    write_csv("router_results.csv", rows)
    write_csv(
        "router_confusion.csv",
        [
            {"expected": e, "predicted": p, "count": c}
            for (e, p), c in sorted(confusion.items())
        ],
    )
    return rows


def eval_text_retrieval(questions: list[dict], ids: dict[str, int]) -> list[dict]:
    print("\n[2] Text retrieval (semantic + hybrid questions)")
    rows = []
    subset = [q for q in questions if q["expected_type"] in ("semantic", "hybrid")]

    for q in subset:
        expected_ids = {ids[n] for n in q["expected"] if n in ids}
        if not expected_ids:
            print(f"  {q['id']}  ! no expected attraction found in the DB, skipped")
            continue

        plan = {
            "query_type": "semantic",
            "semantic_query": q["question"],
            "categories": [],
            "district": "",
        }
        hits = retrieval.retrieve_text(plan, k=5)

        ranked, seen = [], set()
        for h in hits:
            if h["attraction_id"] not in seen:
                seen.add(h["attraction_id"])
                ranked.append(h["attraction_id"])

        rank = first_correct_rank(ranked, expected_ids)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "rank": rank if rank else "",
                "recall@1": int(rank == 1) if rank else 0,
                "recall@3": int(rank <= 3) if rank else 0,
                "recall@5": int(rank <= 5) if rank else 0,
                "reciprocal_rank": round(1 / rank, 4) if rank else 0.0,
                "top_hit": hits[0]["name"] if hits else "",
            }
        )
        print(
            f"  {q['id']}  rank={rank if rank else '-':>3}  "
            f"top='{hits[0]['name'] if hits else '-'}'"
        )

    if rows:
        n = len(rows)
        print(
            f"  Recall@1 {sum(r['recall@1'] for r in rows) / n:.1%}   "
            f"Recall@3 {sum(r['recall@3'] for r in rows) / n:.1%}   "
            f"Recall@5 {sum(r['recall@5'] for r in rows) / n:.1%}   "
            f"MRR {sum(r['reciprocal_rank'] for r in rows) / n:.3f}"
        )
    write_csv("text_retrieval.csv", rows)
    return rows


def eval_image_retrieval() -> list[dict]:
    """Query the index with photographs it has never seen."""
    print("\n[3] Image retrieval (held-out photographs)")
    with psycopg.connect(config.DB_CONN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT i.file_path, i.attraction_id, a.name
            FROM images i JOIN attractions a ON a.id = i.attraction_id
            WHERE i.is_held_out = TRUE
            ORDER BY a.name
            """)
        held_out = cur.fetchall()

    if not held_out:
        print("  no held-out images found -- run ingest.py then embed.py first")
        return []

    rows = []
    for file_path, attraction_id, name in held_out:
        path = config.ROOT / file_path
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        hits = retrieval.retrieve_images({"semantic_query": ""}, image=img, k=5)

        ranked, seen = [], set()
        for h in hits:
            if h["attraction_id"] not in seen:
                seen.add(h["attraction_id"])
                ranked.append(h["attraction_id"])

        rank = first_correct_rank(ranked, {attraction_id})
        rows.append(
            {
                "query_image": file_path,
                "true_attraction": name,
                "predicted": hits[0]["name"] if hits else "",
                "rank": rank if rank else "",
                "top1": int(rank == 1) if rank else 0,
                "top3": int(rank <= 3) if rank else 0,
            }
        )
        print(
            f"  {name:<26} -> {hits[0]['name'] if hits else '-':<26} "
            f"rank={rank if rank else '-'}"
        )

    if rows:
        n = len(rows)
        print(
            f"  top-1 {sum(r['top1'] for r in rows) / n:.1%}   "
            f"top-3 {sum(r['top3'] for r in rows) / n:.1%}   (n={n})"
        )
    write_csv("image_retrieval.csv", rows)
    return rows


MODES = ["structured", "semantic", "hybrid"]


def _ablate_one(q: dict, expected_ids: set[int]) -> dict:
    """Score one question under each retrieval strategy. Runs in a worker thread."""
    row = {
        "id": q["id"],
        "question": q["question"],
        "expected_type": q["expected_type"],
    }
    for mode in MODES:
        # An API failure is not a retrieval failure. Unresolved ones are
        # excluded from the averages rather than counted as misses.
        res, api_failed = None, False
        for attempt in range(3):
            try:
                # skip_generation: the ablation only checks which attractions
                # reached the context, so writing out the final answer is
                # pure waste -- roughly half of ablation's LLM calls, unread.
                res = pipeline.answer(
                    q["question"], force_type=mode, skip_generation=True
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {q['id']} [{mode}] call failed: {str(exc)[:60]}")
                time.sleep(2 * (attempt + 1))
                continue
            if res["sql"] and res["sql"].get("api_error"):
                time.sleep(2 * (attempt + 1))
                api_failed = True
                continue
            api_failed = False
            break

        if res is None or api_failed:
            print(f"  {q['id']} [{mode}] unresolved API error -- excluded")
            row[mode] = ""
            continue

        in_context = set(res["attraction_ids"])
        if res["sql"] and res["sql"].get("rows"):
            in_context |= {r["id"] for r in res["sql"]["rows"] if "id" in r}
        row[mode] = int(bool(in_context & expected_ids))

    return row


def eval_ablation(questions: list[dict], ids: dict[str, int]) -> list[dict]:
    """How often the expected attraction reaches the model, per strategy."""
    print("\n[4] Ablation: SQL-only vs semantic-only vs hybrid")

    scorable = []
    for q in questions:
        expected_ids = {ids[n] for n in q["expected"] if n in ids}
        if expected_ids:
            scorable.append((q, expected_ids))

    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as pool:
        future_to_q = {
            pool.submit(_ablate_one, q, expected_ids): q for q, expected_ids in scorable
        }
        by_id = {}
        for future in as_completed(future_to_q):
            row = future.result()
            by_id[row["id"]] = row
            print(
                f"  {row['id']}  sql={row['structured']}  semantic={row['semantic']}  "
                f"hybrid={row['hybrid']}"
            )

    rows = [by_id[q["id"]] for q, _ in scorable]

    if rows:
        print("\n  context recall:")
        for mode in MODES:
            scored = [r[mode] for r in rows if r[mode] != ""]
            if scored:
                print(
                    f"    {mode:<11} {sum(scored) / len(scored):.1%}  (n={len(scored)})"
                )
    write_csv("ablation.csv", rows)
    return rows


def main() -> None:
    questions = load_questions()
    ids = name_to_id()
    if not ids:
        print("The attractions table is empty -- run src/ingest.py first.")
        return

    # Force these before the thread pools start, so lazy singleton init
    # (the Gemini client, the SQL column-vocabulary cache) doesn't race
    # across the first few concurrent requests.
    generate.client()
    retrieval.column_vocabulary()

    print(
        f"Evaluation set: {len(questions)} questions;  database: {len(ids)} attractions"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    router = eval_router(questions)
    text = eval_text_retrieval(questions, ids)
    images = eval_image_retrieval()
    ablation = eval_ablation(questions, ids)

    summary = []
    if router:
        summary.append(
            {
                "metric": "router accuracy",
                "value": f"{sum(r['correct'] for r in router) / len(router):.1%}",
            }
        )
    if text:
        n = len(text)
        summary += [
            {
                "metric": "text Recall@3",
                "value": f"{sum(r['recall@3'] for r in text) / n:.1%}",
            },
            {
                "metric": "text MRR",
                "value": f"{sum(r['reciprocal_rank'] for r in text) / n:.3f}",
            },
        ]
    if images:
        n = len(images)
        summary += [
            {
                "metric": "image top-1",
                "value": f"{sum(r['top1'] for r in images) / n:.1%}",
            },
            {
                "metric": "image top-3",
                "value": f"{sum(r['top3'] for r in images) / n:.1%}",
            },
        ]
    if ablation:
        for mode in ("structured", "semantic", "hybrid"):
            summary.append(
                {
                    "metric": f"context recall ({mode})",
                    "value": f"{mean_of_scored(ablation, mode):.1%}",
                }
            )

    print("\n" + "=" * 62)
    for s in summary:
        print(f"  {s['metric']:<28} {s['value']}")
    print("=" * 62)
    write_csv("summary.csv", summary)


if __name__ == "__main__":
    sys.exit(main())
