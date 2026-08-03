"""
Streamlit interface.

The expanders show the routing decision, the SQL, the retrieved chunks and the
matched images, since an answer is only worth trusting if the retrieval behind
it can be inspected.

Run:  uv run streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

import config  # noqa: E402
import pipeline  # noqa: E402

st.set_page_config(page_title="Sri Lanka Tourism RAG", page_icon="🇱🇰", layout="wide")

EXAMPLES = {
    "Structured": "Which waterfalls are more than 100 metres tall?",
    "Semantic": "Where can I see colonial era architecture right by the sea?",
    "Image": "Show me photographs of a wide waterfall falling over a rock face",
    "Hybrid": "I want a tall waterfall in Badulla district, how hard is the hike?",
    "Out of scope": "What is the wifi password at the UCSC?",
}

with st.sidebar:
    st.header("About")
    st.markdown(
        "Multimodal RAG over a **PostgreSQL** knowledge base and a "
        "**ChromaDB** vector store holding both text and image embeddings.\n\n"
        "SCS 4203 — Database III, Assignment 2."
    )

    st.subheader("Try an example")
    for label, text in EXAMPLES.items():
        if st.button(label, width="stretch"):
            st.session_state["question"] = text

    st.divider()
    st.caption(
        f"text: `{config.TEXT_MODEL}`  \n"
        f"images: `{config.IMAGE_MODEL}`  \n"
        f"LLM: `{config.GEMINI_MODEL}`"
    )

st.title("Sri Lanka Tourism — Multimodal RAG")

with st.container(border=True):
    text_col, image_col = st.columns([2, 1], gap="medium")

    with text_col:
        question = st.text_area(
            "Ask about waterfalls, beaches, or historical sites in Sri Lanka",
            value=st.session_state.get("question", ""),
            placeholder="e.g. Which waterfalls are more than 100 metres tall?",
            height="stretch",
        )

    with image_col:
        uploaded = st.file_uploader(
            "Or search by photograph",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
    go = st.button("Search", type="primary", width="stretch")

if go and not question.strip() and not uploaded:
    st.warning("Enter a question, or upload a photograph.")

elif go:
    query_image = Image.open(uploaded).convert("RGB") if uploaded else None
    prompt = question.strip() or "What attraction is shown in this photograph?"

    with st.spinner("Retrieving and generating…"):
        try:
            result = pipeline.answer(prompt, image=query_image)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    plan = result["plan"]

    cols = st.columns(4)
    cols[0].metric("Query type", plan["query_type"])
    cols[1].metric("SQL rows", len(result["sql"]["rows"]) if result["sql"] else 0)
    cols[2].metric("Text chunks", len(result["text_hits"]))
    cols[3].metric("Images", len(result["image_hits"]))

    st.subheader("Answer")
    if result["error"]:
        st.error(result["error"])
    else:
        st.markdown(result["answer"] or "_No answer produced._")

    if result["text_hits"]:
        st.markdown("**Sources cited above**")
        for i, hit in enumerate(result["text_hits"], 1):
            link = (
                f" · [{hit['source']}]({hit['source_url']})"
                if hit["source_url"]
                else f" · {hit['source']}"
            )
            with st.expander(
                f"[S{i}]  {hit['name']}  ·  similarity {hit['score']:.3f}"
            ):
                st.markdown(f"*{hit['name']}*{link}")
                st.write(hit["text"])

    if result["image_hits"]:
        st.subheader("Retrieved photographs")
        image_cols = st.columns(min(4, len(result["image_hits"])))
        for col, hit in zip(image_cols, result["image_hits"]):
            path = config.ROOT / hit["file_path"]
            if path.exists():
                col.image(
                    str(path),
                    caption=f"{hit['name']} · similarity {hit['score']:.3f}",
                    width="stretch",
                )

    st.subheader("How this answer was produced")

    with st.expander(f"1 · Routing — classified as **{plan['query_type']}**"):
        st.json(plan)

    if result["sql"]:
        n = len(result["sql"].get("rows", []))
        with st.expander(f"2 · Relational database — {n} rows"):
            if result["sql"].get("sql"):
                st.code(result["sql"]["sql"], language="sql")
            if result["sql"].get("error"):
                st.warning(result["sql"]["error"])
            if result["sql"].get("rows"):
                st.dataframe(result["sql"]["rows"], width="stretch")

    if result["text_hits"]:
        with st.expander(f"3 · Semantic search — {len(result['text_hits'])} chunks"):
            st.dataframe(
                [
                    {
                        "tag": f"S{i}",
                        "attraction": h["name"],
                        "similarity": round(h["score"], 4),
                        "source": h["source"],
                    }
                    for i, h in enumerate(result["text_hits"], 1)
                ],
                width="stretch",
            )

    if result["image_hits"]:
        with st.expander(f"4 · Image search — {len(result['image_hits'])} matches"):
            st.dataframe(
                [
                    {
                        "rank": i,
                        "attraction": h["name"],
                        "similarity": round(h["score"], 4),
                        "file": h["file_path"],
                    }
                    for i, h in enumerate(result["image_hits"], 1)
                ],
                width="stretch",
            )

    with st.expander("5 · Exact context sent to the LLM"):
        st.text(result["context"])
