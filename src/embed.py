"""
Build the two ChromaDB collections from PostgreSQL.

Run:  uv run python src/embed.py
"""

import sys

import chromadb
import psycopg
from PIL import Image

import config  # must come before the sentence_transformers import below

from sentence_transformers import SentenceTransformer  # noqa: E402


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def reset_collection(client, name: str):
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001
        pass
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def embed_text(conn, client) -> int:
    print(f"\nText: loading {config.TEXT_MODEL} ...")
    model = SentenceTransformer(config.TEXT_MODEL)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT d.id, d.chunk_text, d.attraction_id,
                   a.name, a.category, a.district, d.source, d.source_url
            FROM documents d
            JOIN attractions a ON a.id = d.attraction_id
            ORDER BY d.id
            """)
        rows = cur.fetchall()

    if not rows:
        print("  no documents found - run src/ingest.py first")
        return 0, 0

    ids = [f"doc_{r[0]}" for r in rows]
    texts = [r[1] for r in rows]
    metas = [
        {
            "document_id": r[0],
            "attraction_id": r[2],
            "name": r[3],
            "category": r[4],
            "district": r[5] or "",
            "source": r[6],
            "source_url": r[7] or "",
        }
        for r in rows
    ]

    print(f"  encoding {len(texts)} chunks ...")
    vectors = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    coll = reset_collection(client, config.TEXT_COLLECTION)
    coll.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"  '{config.TEXT_COLLECTION}' built: {coll.count()} vectors")
    return coll.count(), len(vectors[0])


def embed_images(conn, client) -> int:
    print(f"\nImages: loading {config.IMAGE_MODEL} ...")
    model = SentenceTransformer(config.IMAGE_MODEL)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT i.id, i.file_path, i.caption, i.attraction_id,
                   a.name, a.category, a.district
            FROM images i
            JOIN attractions a ON a.id = i.attraction_id
            WHERE i.is_held_out = FALSE      - the eval set
            ORDER BY i.id
            """)
        rows = cur.fetchall()

    if not rows:
        print("  no images found - run src/ingest.py first")
        return 0, 0

    ids, metas, pil_images, captions = [], [], [], []
    for img_id, path, caption, attr_id, name, category, district in rows:
        full = config.ROOT / path
        if not full.exists():
            print(f"  ! missing file {path}")
            continue
        try:
            pil_images.append(Image.open(full).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! unreadable {path} ({exc})")
            continue
        ids.append(f"img_{img_id}")
        captions.append(caption or name)
        metas.append(
            {
                "image_id": img_id,
                "attraction_id": attr_id,
                "name": name,
                "category": category,
                "district": district or "",
                "file_path": path,
            }
        )

    print(f"  encoding {len(pil_images)} images ...")
    vectors = model.encode(
        pil_images, batch_size=16, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    coll = reset_collection(client, config.IMAGE_COLLECTION)
    coll.add(ids=ids, embeddings=vectors, documents=captions, metadatas=metas)
    print(f"  '{config.IMAGE_COLLECTION}' built: {coll.count()} vectors")
    return coll.count(), len(vectors[0])


def main() -> None:
    conn = psycopg.connect(config.DB_CONN)
    client = get_client()

    n_text, d_text = embed_text(conn, client)
    n_img, d_img = embed_images(conn, client)

    print("\n" + "=" * 60)
    print(f"text_chunks : {n_text} vectors ({config.TEXT_MODEL}, {d_text}-d)")
    print(f"images      : {n_img} vectors ({config.IMAGE_MODEL}, {d_img}-d)")
    print(f"persisted to: {config.CHROMA_DIR}")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    sys.exit(main())
