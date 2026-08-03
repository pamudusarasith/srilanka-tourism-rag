"""
The RAG pipeline.

    query -> routing -> retrieval -> context integration -> LLM -> answer
"""

from PIL import Image

import config
import generate
import retrieval


def answer(
    question: str,
    image: Image.Image | None = None,
    force_type: str | None = None,
    skip_generation: bool = False,
) -> dict:
    """
    Run the full pipeline.

    `force_type` skips the router. The evaluation uses it to measure each
    retriever on its own. `skip_generation` stops after context integration:
    the ablation only scores which attractions made it into context, so the
    final answer-writing call would otherwise run 108 times and be discarded
    unread.
    """
    if force_type:
        plan = {
            "query_type": force_type,
            "semantic_query": question,
            "categories": [],
            "district": "",
        }
    else:
        plan = retrieval.route(question, has_image=image is not None)

    use_sql, use_text, use_images = retrieval.retrievers_for(plan)

    sql_result = retrieval.retrieve_sql(question) if use_sql else None
    text_hits = retrieval.retrieve_text(plan) if use_text else []
    image_hits = retrieval.retrieve_images(plan, image=image) if use_images else []

    sql_result, text_hits, image_hits, merged_ids = integrate(
        sql_result, text_hits, image_hits
    )

    if skip_generation:
        return {
            "question": question,
            "plan": plan,
            "sql": sql_result,
            "text_hits": text_hits,
            "image_hits": image_hits,
            "attraction_ids": merged_ids,
            "answer": "",
            "context": "",
            "error": None,
        }

    pil_images = []
    for hit in image_hits[: config.MAX_IMAGES_TO_LLM]:
        path = config.ROOT / hit["file_path"]
        if path.exists():
            try:
                pil_images.append(Image.open(path).convert("RGB"))
            except Exception:  # noqa: BLE001
                pass

    result = generate.generate(question, sql_result, text_hits, image_hits, pil_images)

    return {
        "question": question,
        "plan": plan,
        "sql": sql_result,
        "text_hits": text_hits,
        "image_hits": image_hits,
        "attraction_ids": merged_ids,
        "answer": result["answer"],
        "context": result["context"],
        "error": result["error"],
    }


def integrate(
    sql_result: dict | None, text_hits: list[dict], image_hits: list[dict]
) -> tuple[dict | None, list[dict], list[dict], list[int]]:
    """
    Merge the three result sets into one context.

    Ranking by attraction rather than by chunk stops one well-documented site
    filling the window. Every attraction either vector store found is then
    looked up in PostgreSQL, so the model gets exact values alongside the
    prose.
    """
    text_hits = text_hits[: config.TOP_K_TEXT]
    image_hits = image_hits[: config.TOP_K_IMAGE]

    # CLIP similarities sit in a narrow band, so a fixed threshold is no use,
    # but the gap below the best match is. A photo of Sigiriya matches at 1.00
    # while unrelated waterfalls still score ~0.85, and passing those on makes
    # the model describe attractions nobody asked about.
    if image_hits:
        best = image_hits[0]["score"]
        image_hits = [
            h for h in image_hits if best - h["score"] <= config.IMAGE_SCORE_GAP
        ]

    ordered_ids: list[int] = []
    for hit in text_hits + image_hits:
        aid = hit["attraction_id"]
        if aid not in ordered_ids:
            ordered_ids.append(aid)

    # SQL rows are authoritative where present. Otherwise each attraction the
    # vector search found is looked up, so the model still sees exact values.
    if sql_result and sql_result.get("rows"):
        for row in sql_result["rows"]:
            if "id" in row and row["id"] not in ordered_ids:
                ordered_ids.append(row["id"])
    elif ordered_ids:
        rows = retrieval.fetch_attractions(ordered_ids[: config.MAX_SQL_ROWS])
        if rows:
            sql_result = {
                "sql": "- looked up from vector-search hits, not a generated query",
                "rows": rows,
                "error": None,
            }

    return sql_result, text_hits, image_hits, ordered_ids
