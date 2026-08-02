# Demo video script

Target length: 5 minutes. The brief asks the demonstration to show structured,
semantic, image-based and hybrid retrieval, so those four are the spine of the
video. Everything else is optional.

Every query below has been run against the live system with the models actually
configured (`gemma-4-31b-it`, `bge-base-en-v1.5`, `clip-ViT-B-16`) and produces
the result described. Do not substitute queries on the day without testing them
first: several obvious-looking alternatives route differently than you would
expect.

---

## Before you record

1. `docker compose up -d`, then confirm the container is healthy.
2. Start the app and **run one query to warm it up**. The first query of a fresh
   process loads BGE and CLIP and takes 10 to 15 seconds. Every query after that
   is fast. Recording the cold start makes the system look slow.
3. Copy `data/images/devon_falls/03.jpg` to your Desktop and rename it something
   neutral like `my-photo.jpg`. You will upload it in the image section.
4. Browser zoom to about 125% so text is readable in the recording.
5. Close other tabs. Hide bookmarks. Turn off notifications.
6. Have `report/results/summary.csv` open in a second tab for the closing shot.

Avoid asking about entrance fees, swimming safety, surf season or accessibility.
Those four columns are NULL for every row, so the system correctly answers that
it does not know. That is honest behaviour but it is not what you want on camera.

---

## 0:00 – 0:30 · Opening

**Show:** the report's architecture diagram (page 2), full screen.

> This is a multimodal retrieval-augmented generation system for Sri Lankan
> tourist attractions. The knowledge base has 42 attractions across waterfalls,
> beaches and historical sites, stored in two places: structured fields in
> PostgreSQL, and text and image embeddings in ChromaDB.
>
> A query gets classified, sent to whichever stores can answer it, and the
> results are merged into one context before the language model sees them.
> I'll show all four query types.

Trace the flow with your cursor as you say it: router, three stores, context
integration, model.

---

## 0:30 – 1:15 · Structured query

**Show:** the app. Click the "Structured" example button in the sidebar.

**Query:** `Which waterfalls are more than 100 metres tall?`

> First, a question that is purely a filter over a numeric column.

Wait for the answer. Then:

> Five waterfalls, with their heights. The router classified this as
> structured, so only PostgreSQL ran.

**Open expander 2 (Relational database).** This is the important shot.

> This is the SQL the model wrote, and the rows it returned. The language model
> never sees the database directly. It writes a SELECT, we check that it is a
> single read-only statement, then we run it and hand back the rows.

Let the SQL sit on screen for a couple of seconds so it is readable.

---

## 1:15 – 2:00 · Semantic query

**Query:** `Tell me about the ancient rock fortress with frescoes painted on its wall`

> This question names no attraction and no column. There is no "has frescoes"
> field to filter on, so it has to be matched on meaning.

**Result:** Sigiriya, with `[S1]`-style citations in the answer.

**Open expander 3 (Semantic search).**

> Six passages, each with its similarity score and which source it came from,
> Wikipedia or AmazingLanka. The citations in the answer point back to these,
> so every claim can be traced.

Point at the `source` label on a chunk that came from AmazingLanka if one is
in the list.

---

## 2:00 – 3:00 · Image query

**Show:** the sidebar file uploader. Upload `my-photo.jpg` from your Desktop.

**Query:** `What waterfall is this and how hard is the hike?`

> Now an image. This photograph is not in the index. It was deliberately held
> out when the system was built, so this is a picture it has never seen.

Wait for the result.

> It identified Devon Falls, and the top two matches are both Devon Falls.

The router classifies this as `hybrid` rather than `image`, because an uploaded
photograph always forces image retrieval and the text of the question also asks
about trek difficulty. That is expected; do not describe it as an image-only
query on camera.

**Open expander 4 (Image search).**

> These are CLIP similarity scores against 151 indexed photographs. The uploaded
> image is encoded the same way and compared in that shared vector space.

The held-out point is the strongest claim in the whole video. Say it clearly.

---

## 3:00 – 4:00 · Hybrid query

**Query:** `Which UNESCO site in the Central Province should I visit, and what will I actually see there?`

> This one needs both. "UNESCO site in the Central Province" is a filter over
> two columns. "What will I actually see" is not in any column, so it has to
> come from the descriptive text.

**Result:** three UNESCO sites in the Central Province, each with a description
of what is there. Three SQL rows, six text chunks and four images all feed one
answer.

**Open expanders 2 and 3 together**, scrolling between them.

> The SQL retrieved the rows and the vector search retrieved the descriptions,
> and both went into the same context. Look at the metric row at the top:
> three SQL rows, six text chunks and four images in one answer.

Point at the four metric tiles.

**Open expander 5 (Exact context sent to the LLM).**

> This is exactly what the model received. Structured rows at the top, then
> the passages. This is the step that makes it one system instead of three
> separate searches.

---

## 4:00 – 4:30 · Grounding check

**Query:** `What is the wifi password at the Colombo Hilton?`

> One more, deliberately outside the knowledge base.

**Result:** "The knowledge base does not cover that."

> It refuses rather than inventing an answer. For a tourism system that matters,
> because a made-up entrance fee looks exactly like a real one.

---

## 4:30 – 5:00 · Evaluation and close

**Show:** the results table from the report (Evaluation section) or
`report/results/summary.csv`.

> We evaluated on 12 questions. The headline number is the ablation. Using only
> the relational database, the right attraction reaches the model 58% of the
> time. Using only semantic search, 67%. Using both together with image
> retrieval, 100%.
>
> The two stores fail on different questions, which is the point of building it
> this way rather than picking one.

End on the table. No sign-off needed.

---

## Recording notes

- QuickTime (File → New Screen Recording) is enough. Record the browser window,
  not the full desktop.
- Record narration live rather than dubbing. It is faster and sounds more natural.
- If a query is slow or errors on a rate limit, stop, wait a minute, re-record
  that section. Do not leave a visible failure in unless you explain it.
- One take per section is fine. Cut between sections rather than trying to get
  five minutes clean in one pass.
- Export at 1080p and name it `IndexNo1_IndexNo2_Demo.mp4`.

## If you are short on time

The four query types are required. The grounding check and the evaluation
close are not. Cutting those gets you to roughly 3.5 minutes without losing
anything the brief asks for.
