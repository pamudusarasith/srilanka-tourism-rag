"""Query routing and the three retrievers (SQL, text vectors, image vectors)."""

import json
import re

import chromadb
import psycopg
from PIL import Image
from sentence_transformers import SentenceTransformer

import config

STRUCTURED = "structured"
SEMANTIC = "semantic"
IMAGE = "image"
HYBRID = "hybrid"
QUERY_TYPES = (STRUCTURED, SEMANTIC, IMAGE, HYBRID)

# Loaded once per process. Streamlit reruns the script on every interaction,
# which would otherwise reload both models each time.
_text_model: SentenceTransformer | None = None
_image_model: SentenceTransformer | None = None
_chroma: chromadb.ClientAPI | None = None


def text_model() -> SentenceTransformer:
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(config.TEXT_MODEL)
    return _text_model


def image_model() -> SentenceTransformer:
    global _image_model
    if _image_model is None:
        _image_model = SentenceTransformer(config.IMAGE_MODEL)
    return _image_model


def chroma() -> chromadb.ClientAPI:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma


def db():
    return psycopg.connect(config.DB_CONN)


# --- Stage 1: routing ----------------------------------------------------

ROUTER_PROMPT = """You classify questions for a Sri Lanka tourism search system.

The system has three retrievers:
  - SQL over a table of attractions with these columns:
    name, category ('waterfall' | 'beach' | 'heritage'), district, province,
    latitude, longitude, height_m, trekking_difficulty ('easy'|'moderate'|'hard'),
    era, unesco_status (boolean), dress_code, best_season, summary
  - semantic search over descriptive paragraphs about each attraction
  - image similarity search over photographs of each attraction

Classify the question as exactly one of:
  "structured" - answerable purely by filtering/sorting/counting the columns above
                 (e.g. "which waterfalls are over 100 m", "how many UNESCO sites")
  "semantic"   - needs descriptive prose: history, atmosphere, what it is like,
                 what to do, why it is notable
  "image"      - asks to find, identify or show photographs, or refers to a
                 supplied picture
  "hybrid"     - needs a combination: a factual filter AND description, or
                 facts AND pictures

Also produce:
  "semantic_query" - a short phrase describing what to find. Used both for
                     semantic text search and, when no photo is uploaded, as
                     the text query for image search. Almost always needed:
                     leave it empty only if the question is a pure structured
                     filter with nothing left to describe.
  "categories"     - any of ["waterfall","beach","heritage"] the question restricts
                     itself to, else []
  "district"       - a district name if the question names one, else ""

Reply with JSON only, no markdown fence:
{"query_type": "...", "semantic_query": "...", "categories": [], "district": ""}

Question: """


def route(question: str, has_image: bool = False) -> dict:
    """Classify a question and extract retrieval hints."""
    from generate import gemini_json  # imported here to avoid a cycle

    plan = None
    try:
        raw = gemini_json(ROUTER_PROMPT + question)
        if raw:
            plan = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[router] routing failed ({str(exc)[:60]})")

    # Hybrid runs every retriever. Slower, but it never retrieves less than
    # the question needs.
    if not plan or plan.get("query_type") not in QUERY_TYPES:
        plan = {"query_type": HYBRID}

    # setdefault only fills a missing key; the model sometimes returns
    # semantic_query as an empty string, which needs the same fallback.
    plan["semantic_query"] = plan.get("semantic_query") or question
    plan.setdefault("categories", [])
    plan.setdefault("district", "")

    if has_image and plan["query_type"] != IMAGE:
        plan["query_type"] = HYBRID

    return plan


def retrievers_for(plan: dict) -> tuple[bool, bool, bool]:
    """Return (use_sql, use_text, use_images)."""
    t = plan["query_type"]
    return (
        t in (STRUCTURED, HYBRID),
        t in (SEMANTIC, HYBRID),
        t in (IMAGE, HYBRID),
    )


# --- Stage 2a: structured retrieval --------------------------------------

SQL_PROMPT = """You write a single PostgreSQL SELECT statement.

Table: attractions
  id                  integer
  name                text
  category            text   -- 'waterfall' | 'beach' | 'heritage'
  district            text   -- e.g. 'Badulla', 'Kandy', 'Galle'
  province            text
  latitude            double precision
  longitude           double precision
  height_m            numeric   -- waterfalls only, else NULL
  trekking_difficulty text      -- 'easy' | 'moderate' | 'hard', waterfalls only
  era                 text      -- heritage only
  unesco_status       boolean
  dress_code          text      -- heritage only
  best_season         text
  summary             text

Rules:
  - Return ONLY the SQL, no explanation and no markdown fence.
  - It must be a single SELECT. Never INSERT, UPDATE, DELETE, DROP or ALTER.
  - Always include id, name and category in the selected columns.
  - Add LIMIT {limit} unless the question implies fewer rows.
  - If the question cannot be answered from these columns, return exactly: NONE

The exact values stored in the low-cardinality columns are listed below. Use
them verbatim. The database stores 'Central', not 'Central Province'.
{vocabulary}
Question: """

_vocab_cache: str | None = None


def column_vocabulary() -> str:
    """
    Distinct values of the low-cardinality columns, for the SQL prompt.

    Without them the model invents plausible ones, such as `province =
    'Central Province'` against a column storing 'Central'. That SQL is valid
    and matches nothing.
    """
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache

    lines = []
    try:
        with db() as conn, conn.cursor() as cur:
            for col in (
                "category",
                "province",
                "district",
                "trekking_difficulty",
                "era",
                "best_season",
            ):
                cur.execute(
                    f"SELECT DISTINCT {col} FROM attractions "
                    f"WHERE {col} IS NOT NULL ORDER BY 1"
                )
                values = [str(r[0]) for r in cur.fetchall()]
                if values:
                    lines.append(f"  {col}: {', '.join(repr(v) for v in values)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[sql] could not read column vocabulary ({str(exc)[:60]})")
        return ""

    _vocab_cache = "\n".join(lines) + "\n"
    return _vocab_cache


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|create|copy)\b",
    re.IGNORECASE,
)


def is_safe_select(sql: str) -> bool:
    """Reject anything that is not a single read-only SELECT."""
    s = sql.strip().rstrip(";").strip()
    if not s.lower().startswith("select"):
        return False
    if ";" in s:
        return False
    return not _FORBIDDEN.search(s)


def retrieve_sql(question: str, limit: int = config.MAX_SQL_ROWS) -> dict:
    """Translate the question to SQL, run it, and return the rows plus the SQL used."""
    from generate import gemini_json

    out = {"sql": None, "rows": [], "error": None, "api_error": False}
    prompt = SQL_PROMPT.format(limit=limit, vocabulary=column_vocabulary())
    try:
        sql = gemini_json(prompt + question, expect_json=False)
    except Exception as exc:  # noqa: BLE001
        # Flagged separately so the evaluation can tell an empty result from
        # a call that never completed.
        out["error"] = f"SQL generation failed: {str(exc)[:80]}"
        out["api_error"] = True
        return out

    if not sql:
        out["error"] = "no SQL generated"
        return out

    sql = sql.strip().strip("`")
    sql = re.sub(r"^sql\s*", "", sql, flags=re.IGNORECASE).strip()
    if sql.upper().startswith("NONE"):
        out["error"] = "question is not answerable from the structured columns"
        return out

    if not is_safe_select(sql):
        out["error"] = "generated statement rejected by the safety check"
        out["sql"] = sql
        return out

    out["sql"] = sql
    try:
        with db() as conn, conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            out["rows"] = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"SQL execution failed: {str(exc)[:120]}"
    return out


def fetch_attractions(ids: list[int]) -> list[dict]:
    """Expand attraction ids into full rows, preserving the given order."""
    if not ids:
        return []
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM attractions WHERE id = ANY(%s)", (list(ids),))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    order = {aid: i for i, aid in enumerate(ids)}
    rows.sort(key=lambda r: order.get(r["id"], 999))
    return rows


# --- Stage 2b and 2c: vector retrieval -----------------------------------


def _metadata_filter(plan: dict) -> dict | None:
    """
    Chroma `where` clause from the router's hints.

    Structured knowledge narrowing the semantic search. "temples near Kandy"
    searches only Kandy instead of hoping the embedding encodes the district.
    """
    clauses = []
    cats = plan.get("categories") or []
    if cats:
        clauses.append({"category": {"$in": cats}})
    if plan.get("district"):
        clauses.append({"district": plan["district"]})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve_text(plan: dict, k: int = config.TOP_K_TEXT) -> list[dict]:
    query = plan.get("semantic_query") or ""
    if not query.strip():
        return []

    coll = chroma().get_collection(config.TEXT_COLLECTION)
    vec = text_model().encode([query], normalize_embeddings=True).tolist()

    where = _metadata_filter(plan)
    res = coll.query(query_embeddings=vec, n_results=k, where=where)
    if where and not res["ids"][0]:
        res = coll.query(query_embeddings=vec, n_results=k)

    hits = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        hits.append(
            {
                "text": res["documents"][0][i],
                "score": 1.0 - res["distances"][0][i],
                "attraction_id": meta["attraction_id"],
                "name": meta["name"],
                "source": meta.get("source", "wikipedia"),
                "source_url": meta.get("source_url", ""),
            }
        )
    return hits


def retrieve_images(
    plan: dict, image: Image.Image | None = None, k: int = config.TOP_K_IMAGE
) -> list[dict]:
    """
    Image-to-image search when given a photo, text-to-image otherwise.

    One collection serves both, since CLIP shares a vector space between the
    two modalities.
    """
    coll = chroma().get_collection(config.IMAGE_COLLECTION)
    model = image_model()

    if image is not None:
        vec = model.encode([image], normalize_embeddings=True).tolist()
    else:
        query = plan.get("semantic_query") or ""
        if not query.strip():
            return []
        vec = model.encode([query], normalize_embeddings=True).tolist()

    where = _metadata_filter(plan) if image is None else None
    res = coll.query(query_embeddings=vec, n_results=k, where=where)
    if where and not res["ids"][0]:
        res = coll.query(query_embeddings=vec, n_results=k)

    hits = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        hits.append(
            {
                "caption": res["documents"][0][i],
                "score": 1.0 - res["distances"][0][i],
                "attraction_id": meta["attraction_id"],
                "name": meta["name"],
                "file_path": meta["file_path"],
            }
        )
    return hits
