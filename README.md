# Multimodal RAG for Sri Lanka Tourism

SCS 4203 — Database III, Assignment 2.

A Retrieval-Augmented Generation system that answers natural-language questions
about Sri Lankan tourist attractions by combining a **relational database**
(PostgreSQL) with a **vector database** (ChromaDB, holding both text and image
embeddings), then passing the assembled context to an LLM.

Covers three attraction categories: **waterfalls**, **beaches**, and
**historical sites & temples**.

## Query types supported

| Type | Example | Path through the system |
|---|---|---|
| Structured | *"Which waterfalls are over 100 m tall?"* | PostgreSQL |
| Semantic | *"Where can I see colonial-era architecture by the sea?"* | ChromaDB `text_chunks` |
| Image | *(upload a photo)* *"Where is this?"* | ChromaDB `images` (CLIP) |
| Hybrid | *"Show me a tall waterfall in Badulla and tell me how hard the hike is"* | all three, fused |

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker (for PostgreSQL)
- A Google Gemini API key ([free tier](https://aistudio.google.com/apikey))

## Installation

```bash
git clone https://github.com/pamudusarasith/srilanka-tourism-rag.git && cd srilanka-tourism-rag
uv sync                       # creates .venv and installs dependencies
cp .env.example .env          # then paste your GEMINI_API_KEY into .env
docker compose up -d          # starts PostgreSQL on host port 5432
```

## Building the knowledge base

```bash
uv run python src/ingest.py   # ~15 min: schema, web sources, PostgreSQL
uv run python src/embed.py    # ~3 min: builds both ChromaDB collections
```

`ingest.py` is destructive by design: it re-applies `src/schema.sql` and
rebuilds the database from `data/seeds/attractions.csv` on every run. Each
attraction is committed independently, so a network failure does not discard earlier work.
`embed.py` rebuilds both vector collections from PostgreSQL, 
so the vector index can always be regenerated from the relational store.

### Text sources

Two sources are combined, because neither is sufficient alone:

- **Wikipedia** — reliable and CC BY-SA licensed, but many Sri Lankan
  waterfalls have only stub articles (Bambarakanda Falls is 131 words).
- **[AmazingLanka](https://amazinglanka.com/wp/)** — a Sri Lankan travel
  encyclopaedia with much richer descriptions of access routes, trail
  conditions and history. Retrieved through its public WordPress REST API.

Every chunk in `documents` records which source it came from, so any
retrieved passage can be traced to its page. Images come from Wikimedia
Commons only: AmazingLanka's photographs are copyrighted, so its text is
attributed and linked but its images are never redistributed.

## Running

```bash
uv run streamlit run app.py       # web interface
uv run python src/evaluate.py     # evaluation metrics
```

## Project layout

```
├── docker-compose.yml            PostgreSQL 16
├── pyproject.toml                dependencies, Python 3.12 pin
├── data/
│   ├── seeds/attractions.csv     curated structured fields (hand-maintained)
│   ├── seeds/eval_questions.csv  evaluation questions with expected answers
│   ├── images/                   downloaded photographs
│   └── chroma/                   persistent vector store
├── src/
│   ├── config.py                 all tunables
│   ├── schema.sql                relational schema
│   ├── sources.py                Wikipedia / AmazingLanka / Commons clients
│   ├── ingest.py                 web sources -> PostgreSQL
│   ├── embed.py                  PostgreSQL -> ChromaDB (text + image collections)
│   ├── retrieval.py              router + the three retrievers
│   ├── generate.py               LLM prompting and answer generation
│   ├── pipeline.py               the five pipeline stages
│   └── evaluate.py               retrieval / router / ablation metrics
├── report/results/               evaluation output (CSV)
└── app.py                        Streamlit interface
```

## Sources

- **Text and coordinates** — [Wikipedia](https://en.wikipedia.org/), via the MediaWiki API, CC BY-SA.
- **Additional descriptive text** — [AmazingLanka](https://amazinglanka.com/wp/), via its WordPress REST API. Used for non-commercial academic purposes; no images are taken from this source.
- **Photographs** — [Wikimedia Commons](https://commons.wikimedia.org/), CC BY-SA / CC BY. Per-image licence and author are stored in the `images` table.
- **Structured fields** (entrance fees, trekking difficulty, dress codes) — curated by hand in `data/seeds/attractions.csv`.
- **Models** — `BAAI/bge-base-en-v1.5` (text embeddings), `clip-ViT-B-16` (image embeddings), `gemma-4-31b-it` (generation).
