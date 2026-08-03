DROP TABLE IF EXISTS documents;
DROP TABLE IF EXISTS images;
DROP TABLE IF EXISTS attractions;

CREATE TABLE attractions (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    category            TEXT NOT NULL
                        CHECK (category IN ('waterfall', 'beach', 'heritage')),
    district            TEXT,
    province            TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,

    height_m            NUMERIC(6,1),
    trekking_difficulty TEXT CHECK (trekking_difficulty IN
                        ('easy', 'moderate', 'hard') OR trekking_difficulty IS NULL),

    era                 TEXT,
    unesco_status       BOOLEAN DEFAULT FALSE,
    dress_code          TEXT,

    best_season         TEXT,
    summary             TEXT,
    wiki_url            TEXT
);

CREATE INDEX idx_attractions_category ON attractions (category);
CREATE INDEX idx_attractions_district ON attractions (district);
CREATE INDEX idx_attractions_height   ON attractions (height_m);

CREATE TABLE images (
    id            SERIAL PRIMARY KEY,
    attraction_id INTEGER NOT NULL REFERENCES attractions(id) ON DELETE CASCADE,
    file_path     TEXT NOT NULL,
    caption       TEXT,
    source_url    TEXT,
    license       TEXT,
    is_held_out   BOOLEAN DEFAULT FALSE   -- kept out of the index, for evaluation
);

CREATE INDEX idx_images_attraction ON images (attraction_id);

CREATE TABLE documents (
    id            SERIAL PRIMARY KEY,
    attraction_id INTEGER NOT NULL REFERENCES attractions(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    chunk_text    TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'wikipedia'
                  CHECK (source IN ('wikipedia', 'amazinglanka')),
    source_url    TEXT,
    UNIQUE (attraction_id, chunk_index)
);

CREATE INDEX idx_documents_attraction ON documents (attraction_id);
CREATE INDEX idx_documents_source     ON documents (source);
