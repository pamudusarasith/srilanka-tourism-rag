"""Answer generation with Google Gemini."""

import re
import time

from google import genai
from google.genai import types

import config

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "429",
            "resource_exhausted",
            "rate limit",
            "quota",
            "unavailable",
            "503",
        )
    )


def call_with_retry(**kwargs):
    """
    Call Gemini, backing off on rate limits.

    The free tier caps requests per minute and the evaluation sends them in
    bursts. Without this, refusals show up as retrieval failures and skew the
    measured numbers.
    """
    delay = config.LLM_RETRY_BASE_DELAY
    last: Exception | None = None
    for _ in range(config.LLM_RETRIES):
        try:
            return client().models.generate_content(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_rate_limit(exc):
                raise
            time.sleep(delay)
            delay *= 2
    raise last


def gemini_json(prompt: str, expect_json: bool = True) -> str | None:
    """Short deterministic call, for routing and SQL generation."""
    resp = call_with_retry(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    text = (resp.text or "").strip()
    if not text:
        # Raised rather than returned so an exhausted token budget cannot be
        # mistaken for a routing or SQL-generation decision.
        raise RuntimeError("model returned no text")
    if expect_json:
        text = _unfence(text)
        m = re.search(r"\{.*\}", text, re.S)
        return m.group(0) if m else text
    return _unfence(text)


def _unfence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


SYSTEM_PROMPT = """You are a knowledgeable guide to tourist attractions in Sri Lanka.

Answer the user's question using ONLY the context below. The context comes
from a database of attractions, from descriptive passages, and from
photographs retrieved for this question.

Rules:
- Use only facts present in the context. Never invent heights, fees, distances
  or dates.
- Cite descriptive passages with their tags, like [S1] or [S2], immediately
  after the sentence that uses them.
- When the structured table answers the question, state the figures plainly.
- If the context does not contain enough information, say so directly:
  "The knowledge base does not cover that." Do not guess and do not fall back
  on general knowledge about Sri Lanka.
- If photographs were retrieved, refer to what they show, but do not claim to
  see details that are not described.
- Be concise: a short paragraph or a short list, not an essay.
"""


def build_context(
    sql_result: dict | None, text_hits: list[dict], image_hits: list[dict]
) -> str:
    """Assemble the retrieved material into one prompt block with [S1] labels."""
    parts = []

    if sql_result and sql_result.get("rows"):
        parts.append("## Structured data (from the relational database)\n")
        parts.append(f"Query used: {sql_result.get('sql', '')}\n")
        for row in sql_result["rows"]:
            fields = ", ".join(
                f"{k}={v}"
                for k, v in row.items()
                if v is not None and k not in ("summary",)
            )
            parts.append(f"- {fields}")
        parts.append("")

    if text_hits:
        parts.append("## Descriptive passages\n")
        for i, hit in enumerate(text_hits, start=1):
            parts.append(f"[S{i}] ({hit['name']}, source: {hit['source']})")
            parts.append(hit["text"])
            parts.append("")

    if image_hits:
        parts.append("## Photographs retrieved (shown to you as images below)\n")
        for i, hit in enumerate(image_hits, start=1):
            parts.append(f"[IMG{i}] {hit['name']} - {hit['caption']}")
        parts.append("")

    if not parts:
        parts.append("(no context retrieved)")

    return "\n".join(parts)


def generate(
    question: str,
    sql_result: dict | None,
    text_hits: list[dict],
    image_hits: list[dict],
    images: list | None = None,
) -> dict:
    """Answer from the context. Photographs go to the model as image parts."""
    context = build_context(sql_result, text_hits, image_hits)

    parts: list = [
        SYSTEM_PROMPT,
        f"\n# Context\n{context}\n",
        f"\n# Question\n{question}\n",
    ]
    for img in (images or [])[: config.MAX_IMAGES_TO_LLM]:
        parts.append(img)

    try:
        resp = call_with_retry(
            model=config.GEMINI_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        answer = (resp.text or "").strip()
        error = None if answer else "Model returned no text"
    except Exception as exc:  # noqa: BLE001
        answer = ""
        error = f"Generation failed: {str(exc)[:200]}"

    return {"answer": answer, "context": context, "error": error}
