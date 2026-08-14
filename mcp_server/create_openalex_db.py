#!/usr/bin/env python3
"""Create the OpenAlex SQLite subset from the cached raw JSONL.

This module is the executable half of Section 2 of the "05 - MCP Server"
notebook. The notebook explains every design decision in prose and imports
the pieces defined here one at a time; this file holds the actual logic so
the exact same build can be rerun outside the notebook:

    python create_openalex_db.py [raw_path] [db_path]

Why the split? Two reasons:

1. **Reproducibility.** A database that can only be rebuilt by re-executing
   a notebook top-to-bottom is fragile. A standalone script with a CLI entry
   point can be run from a Makefile, a CI job, or a colleague's shell.
2. **Testability.** The notebook smoke-tests `reconstruct_abstract` and
   `parse_work` with asserts on real data. That is only possible because
   parsing is importable, separate from I/O.

The pipeline this file implements, end to end:

    data/openalex_raw.jsonl.gz      (gzipped JSON Lines, one work per line,
        |                            exactly as fetched from the API --
        |                            acquisition is cached separately so
        |                            modeling decisions are re-runnable
        v                            without re-downloading)
    parse_work()                    (one nested JSON document -> flat rows
        |                            grouped by target table)
        v
    build_database()                (two-pass bulk load into SQLite)
        |
        v
    ensure_fts()                    (BM25 full-text index over titles and
        |                            abstracts, for the search_works tool)
        v
    data/openalex.db                (single-file, zero-config, opened
                                     read-only by the MCP server)

Every stage is cache-guarded: if its output already exists it is skipped,
and deleting the artifact is the explicit "rebuild" signal. This makes the
whole pipeline idempotent -- safe to rerun after a crash or a config edit.
"""

import gzip
import json
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema
#
# The schema deliberately mirrors OpenAlex's own entity graph instead of
# flattening everything into one wide table:
#
#    sources <------ works ------> topics      (via work_topics)
#                   ^  |  ^
#                   |  |  +------- works       (via citations)
#    authorships ---+  |
#         |            |
#      authors      affiliations -> institutions
#
# Rationale: the whole point of handing this database to an agent is that
# answering real questions requires *composing* entities with joins. A
# pre-joined flat table would do the model's reasoning for it and freeze
# the space of askable questions.
#
# Design decisions encoded below:
#
# * TEXT primary keys, shortened. OpenAlex IDs are URLs
#   ("https://openalex.org/W2741809807"); we store only the trailing key
#   ("W2741809807"). Joins stay readable, indexes stay small, and the full
#   URL is trivially reconstructible by prepending the prefix.
# * `authorships` and `affiliations` are *separate* link tables. An
#   authorship (author x work) can carry multiple institutions, and
#   institution resolution can fail independently of author resolution.
#   Collapsing them into one table would lose real information; this is
#   the many-to-many structure the data actually has.
# * `citations` is an edge table filtered to the subset: we only keep
#   edges whose target is also one of our works, giving a closed-world
#   citation graph that is honest about its own boundary (the notebook's
#   EDA section 3.6 makes that boundary visible).
# * One statement per table, and an index on every foreign key we expect
#   to join through -- without them, the agent's multi-hop queries would
#   degrade to full table scans.
# * PRAGMA foreign_keys = ON makes SQLite actually enforce the REFERENCES
#   clauses. It is OFF by default for historical compatibility reasons,
#   and leaving it off is how orphan rows are born.
# ---------------------------------------------------------------------------

DDL = """
PRAGMA foreign_keys = ON;

-- Venues: journals, repositories (arXiv!), conferences. `type` matters
-- because an agent asked about "journals" should not count arXiv as one.
CREATE TABLE IF NOT EXISTS sources (
    id            TEXT PRIMARY KEY,
    issn_l        TEXT,
    display_name  TEXT,
    type          TEXT
);

-- The central entity. `cited_by_count` is OpenAlex's *global* citation
-- count; in-subset citations live in the `citations` edge table -- two
-- different numbers answering two different questions.
CREATE TABLE IF NOT EXISTS works (
    id                TEXT PRIMARY KEY,
    doi               TEXT,
    title             TEXT,
    publication_year  INTEGER,
    publication_date  TEXT,
    type              TEXT,
    cited_by_count    INTEGER,
    source_id         TEXT REFERENCES sources(id),
    abstract          TEXT
);

CREATE TABLE IF NOT EXISTS authors (
    id            TEXT PRIMARY KEY,
    orcid         TEXT,
    display_name  TEXT
);

CREATE TABLE IF NOT EXISTS institutions (
    id            TEXT PRIMARY KEY,
    ror           TEXT,
    display_name  TEXT,
    country_code  TEXT,
    type          TEXT
);

-- OpenAlex's topic taxonomy: topic -> subfield -> field -> domain. We
-- denormalize the three ancestor names into the topic row (they are tiny
-- and never queried independently).
CREATE TABLE IF NOT EXISTS topics (
    id            TEXT PRIMARY KEY,
    display_name  TEXT,
    subfield      TEXT,
    field         TEXT,
    domain        TEXT
);

-- Link table: which author wrote which work, and in what position.
-- `is_corresponding` is stored as 0/1 (SQLite has no BOOLEAN type).
CREATE TABLE IF NOT EXISTS authorships (
    work_id          TEXT NOT NULL REFERENCES works(id),
    author_id        TEXT NOT NULL REFERENCES authors(id),
    author_position  TEXT,
    is_corresponding INTEGER,
    PRIMARY KEY (work_id, author_id)
);

-- Link table: which institution an author was at *for a given work*.
-- Kept separate from authorships because one authorship can carry
-- several institutions (or none, when resolution fails).
CREATE TABLE IF NOT EXISTS affiliations (
    work_id        TEXT NOT NULL,
    author_id      TEXT NOT NULL,
    institution_id TEXT NOT NULL REFERENCES institutions(id),
    PRIMARY KEY (work_id, author_id, institution_id)
);

-- Link table: works x topics, with the classifier's confidence score.
-- `is_primary` flags the single topic OpenAlex considers the best fit.
CREATE TABLE IF NOT EXISTS work_topics (
    work_id     TEXT NOT NULL REFERENCES works(id),
    topic_id    TEXT NOT NULL REFERENCES topics(id),
    score       REAL,
    is_primary  INTEGER,
    PRIMARY KEY (work_id, topic_id)
);

-- Closed-world citation graph: only edges where *both* endpoints are in
-- our subset. References pointing outside the slice are dropped.
CREATE TABLE IF NOT EXISTS citations (
    citing_id  TEXT NOT NULL REFERENCES works(id),
    cited_id   TEXT NOT NULL REFERENCES works(id),
    PRIMARY KEY (citing_id, cited_id)
);

-- One index per join path the EDA and the MCP server actually use.
-- (The primary keys above already index the "forward" direction; these
-- cover the reverse lookups: works by year/venue, works by author, works
-- by institution, works by topic, and who-cites-this-work.)
CREATE INDEX IF NOT EXISTS idx_works_year        ON works(publication_year);
CREATE INDEX IF NOT EXISTS idx_works_source      ON works(source_id);
CREATE INDEX IF NOT EXISTS idx_authorships_auth  ON authorships(author_id);
CREATE INDEX IF NOT EXISTS idx_affil_inst        ON affiliations(institution_id);
CREATE INDEX IF NOT EXISTS idx_affil_auth        ON affiliations(author_id);
CREATE INDEX IF NOT EXISTS idx_wt_topic          ON work_topics(topic_id);
CREATE INDEX IF NOT EXISTS idx_citations_cited   ON citations(cited_id);
"""


# ---------------------------------------------------------------------------
# Parsing: one raw OpenAlex record -> per-table rows
#
# Parsing is kept completely separate from insertion: parse_work() touches
# no database, returns plain tuples, and can therefore be unit-tested on a
# single record (the notebook does exactly that). The loader below is then
# free to batch, reorder, and wrap everything in transactions without the
# parsing logic knowing or caring.
# ---------------------------------------------------------------------------

def reconstruct_abstract(inverted_index):
    """Invert OpenAlex's {word: [positions]} index back into plain text.

    OpenAlex does not distribute abstracts as prose -- publisher agreements
    prevent redistributing the verbatim text, but an *inverted index*
    (each word mapped to the list of positions where it occurs) is
    permitted. Since the index preserves every position, the original
    word sequence is fully recoverable:

        {"scaling": [0, 3], "laws": [1], "of": [2]}
            -> [(0, "scaling"), (3, "scaling"), (1, "laws"), (2, "of")]
            -> sort by position
            -> "scaling laws of scaling"

    We need the linear text back because FTS5 (and BM25 ranking) operate
    on running text, not on position maps.

    Returns None for missing/empty input -- roughly a third of works have
    no abstract at all, and NULL is the honest representation of that
    (the notebook's missing-data audit quantifies it).
    """
    if not inverted_index:
        return None
    positioned = [(pos, word)
                  for word, positions in inverted_index.items()
                  for pos in positions]
    return " ".join(word for _, word in sorted(positioned))


def short_id(url):
    """'https://openalex.org/W123' -> 'W123' (also handles None).

    OpenAlex IDs are full URLs. Storing just the trailing key keeps join
    columns and indexes ~4x smaller and query output readable; the URL is
    reconstructible by prepending 'https://openalex.org/'. Nested
    references are frequently missing in the raw data, so None must pass
    through safely rather than raise.
    """
    return url.rsplit("/", 1)[-1] if url else None


def parse_work(rec):
    """Flatten one raw OpenAlex work record into rows grouped by table.

    Returns a dict with one key per target table:

        {"work": (single tuple, the works row),
         "sources": [...], "authors": [...], "institutions": [...],
         "topics": [...], "authorships": [...], "affiliations": [...],
         "work_topics": [...], "refs": [(citing_id, cited_id), ...]}

    Entity rows (sources/authors/institutions/topics) are emitted every
    time they are seen -- the same author appears on many works -- and
    deduplication is deferred to the loader's INSERT OR IGNORE. That
    keeps this function stateless and independently testable.

    "refs" are raw citation edges pointing anywhere in OpenAlex; whether
    the target exists in our subset is unknowable record-by-record, so
    filtering to internal edges is deferred to build_database()'s second
    pass.

    Defensive `or {}` / `or []` guards appear throughout because real
    OpenAlex records omit or null nearly any nested field: works without
    a primary_location, authorships without a resolved author,
    institutions without an ID. A record missing data should degrade to
    fewer rows, never crash the load.
    """
    wid = short_id(rec["id"])
    out = {"sources": [], "authors": [], "institutions": [], "topics": [],
           "authorships": [], "affiliations": [], "work_topics": [],
           "refs": []}

    # The venue lives inside primary_location.source; either level may be
    # missing (preprints without a resolved venue, etc.).
    loc = rec.get("primary_location") or {}
    src = loc.get("source") or {}
    source_id = short_id(src.get("id"))
    if source_id:
        out["sources"].append((source_id, src.get("issn_l"),
                               src.get("display_name"), src.get("type")))

    out["work"] = (wid, rec.get("doi"), rec.get("title"),
                   rec.get("publication_year"), rec.get("publication_date"),
                   rec.get("type"), rec.get("cited_by_count"), source_id,
                   reconstruct_abstract(rec.get("abstract_inverted_index")))

    # Each authorship fans out into: an author row, an authorship link
    # row, and zero or more (institution row + affiliation link row).
    for auth in rec.get("authorships") or []:
        a = auth.get("author") or {}
        aid = short_id(a.get("id"))
        if not aid:
            continue                     # unresolved author: nothing to link
        out["authors"].append((aid, a.get("orcid"), a.get("display_name")))
        out["authorships"].append((wid, aid, auth.get("author_position"),
                                   int(bool(auth.get("is_corresponding")))))
        for inst in auth.get("institutions") or []:
            iid = short_id(inst.get("id"))
            if not iid:
                continue                 # unresolved institution: skip link
            out["institutions"].append((iid, inst.get("ror"),
                                        inst.get("display_name"),
                                        inst.get("country_code"),
                                        inst.get("type")))
            out["affiliations"].append((wid, aid, iid))

    # `topics` lists every assigned topic with a score; `primary_topic`
    # singles one out. We store the flag rather than duplicating the row.
    primary = short_id((rec.get("primary_topic") or {}).get("id"))
    for t in rec.get("topics") or []:
        tid = short_id(t.get("id"))
        if not tid:
            continue
        out["topics"].append((tid, t.get("display_name"),
                              (t.get("subfield") or {}).get("display_name"),
                              (t.get("field") or {}).get("display_name"),
                              (t.get("domain") or {}).get("display_name")))
        out["work_topics"].append((wid, tid, t.get("score"),
                                   int(tid == primary)))

    out["refs"] = [(wid, short_id(r))
                   for r in rec.get("referenced_works") or []]
    return out


# ---------------------------------------------------------------------------
# Loading: two-pass bulk load (entities first, then in-subset citations)
#
# Why two passes? `referenced_works` points into all of OpenAlex, but we
# only want citation edges internal to our subset -- and we cannot know
# which targets exist locally until we have read every record once. So
# pass 1 loads all entities and link tables while collecting the set of
# work IDs and the raw reference list; pass 2 filters the references to
# internal edges and inserts them.
# ---------------------------------------------------------------------------

# INSERT OR IGNORE is the deduplication strategy for entities: the same
# author/institution/source/topic is emitted by parse_work() for every
# work it appears on, and the primary-key conflict silently drops the
# duplicates -- first write wins, which is fine because entity attributes
# are identical across records.
INSERTS = {
    "sources":      "INSERT OR IGNORE INTO sources      VALUES (?,?,?,?)",
    "authors":      "INSERT OR IGNORE INTO authors      VALUES (?,?,?)",
    "institutions": "INSERT OR IGNORE INTO institutions VALUES (?,?,?,?,?)",
    "topics":       "INSERT OR IGNORE INTO topics       VALUES (?,?,?,?,?)",
    "authorships":  "INSERT OR IGNORE INTO authorships  VALUES (?,?,?,?)",
    "affiliations": "INSERT OR IGNORE INTO affiliations VALUES (?,?,?)",
    "work_topics":  "INSERT OR IGNORE INTO work_topics  VALUES (?,?,?,?)",
}


def build_database(raw_path, db_path):
    """Build the normalized database from the raw gzip JSONL cache.

    Bulk-load choices, in order of impact:

    * **One big transaction per pass** (the `with conn:` blocks). SQLite
      commits are fsync-bound; wrapping the whole load in a single
      transaction instead of autocommitting per row is the difference
      between seconds and hours.
    * **`PRAGMA synchronous = OFF`**: skip fsyncs entirely during the
      load. Crash-safety matters for a live server, not for a rebuildable
      artifact -- if the machine dies mid-build we just delete the file
      and rerun. The pragma is connection-scoped, so the MCP server's own
      (read-only) connections are unaffected.
    * **Parents before children**: `works.source_id` REFERENCES
      sources(id) and foreign keys are enforced, so each record's source
      row is inserted before its work row.
    * **`ANALYZE` at the end** populates the query-planner statistics for
      our indexes; without it SQLite guesses cardinalities and can pick
      bad join orders on exactly the multi-hop queries we built the
      indexes for.

    Destructive by design: any existing db_path is deleted first. The
    cache guard lives one level up, in create_database().
    """
    db_path = Path(db_path)
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(DDL)
    conn.execute("PRAGMA synchronous = OFF")      # bulk-load mode

    work_ids, all_refs = set(), []
    with conn, gzip.open(raw_path, "rt") as fh:   # pass 1: everything
        for i, line in enumerate(fh):             # except citations
            row = parse_work(json.loads(line))
            # parents before children: works.source_id references
            # sources(id), so the source row must exist first
            if row["sources"]:
                conn.executemany(INSERTS["sources"], row["sources"])
            conn.execute(
                "INSERT OR IGNORE INTO works VALUES (?,?,?,?,?,?,?,?,?)",
                row["work"])
            for table, stmt in INSERTS.items():
                if table != "sources" and row[table]:
                    conn.executemany(stmt, row[table])
            work_ids.add(row["work"][0])
            all_refs.extend(row["refs"])
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1:,} works loaded")

    # pass 2: keep only citation edges whose target is inside the subset.
    # This is what makes the citation graph "closed-world" -- and the
    # printed percentage makes the boundary visible: for a topical slice,
    # most references point outside it.
    internal = [(a, b) for a, b in all_refs if b in work_ids]
    with conn:                                    # pass 2: internal edges
        conn.executemany(
            "INSERT OR IGNORE INTO citations VALUES (?,?)", internal)
    print(f"citations: kept {len(internal):,} internal edges "
          f"of {len(all_refs):,} references "
          f"({len(internal) / max(len(all_refs), 1):.1%})")

    conn.execute("ANALYZE")   # query-planner statistics for our indexes
    conn.close()


def ensure_fts(conn):
    """Build the works_fts FTS5 table if missing.

    Why FTS at all: agents are bad at guessing LIKE patterns against a
    vocabulary they have never seen ('%chinchila%' misspells its way to
    zero rows). FTS5 gives BM25-ranked search over titles and abstracts,
    which becomes the server's `search_works` tool -- the model's entry
    point *into* the graph, after which it navigates by joins.

    Implementation choice: a standalone FTS table that duplicates title
    and abstract, keyed by work_id. FTS5's external-content mode would
    avoid the duplication, but requires triggers to stay in sync -- and
    our data never changes after the build, so we take the simplest
    correct thing at the cost of some disk. `work_id UNINDEXED` stores
    the key for join-back without wasting index space tokenizing it.

    Returns True if the table was built, False on cache hit (drop the
    table to force a rebuild). Raises sqlite3.OperationalError if this
    Python's SQLite was compiled without FTS5 -- callers should catch it
    and degrade to LIKE-based search rather than fail the whole build.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'works_fts'").fetchone()
    if exists:
        return False
    conn.executescript("""
        CREATE VIRTUAL TABLE works_fts
        USING fts5(work_id UNINDEXED, title, abstract);
    """)
    conn.execute("""INSERT INTO works_fts
                    SELECT id, title, abstract FROM works""")
    conn.commit()
    return True


def create_database(raw_path, db_path):
    """Cache-guarded end-to-end build: relational load + FTS index.

    The guard mirrors the notebook's convention for every expensive step:
    if the artifact exists, skip and say how to rebuild (delete it).
    The FTS step runs even on a cache hit because it has its own guard
    inside ensure_fts() -- an older database built before FTS was added
    gets its index backfilled here.
    """
    db_path = Path(db_path)
    if db_path.exists():
        print(f"cache hit: {db_path} - delete it to rebuild")
    else:
        build_database(raw_path, db_path)
    conn = sqlite3.connect(db_path)
    try:
        if ensure_fts(conn):
            print("works_fts built")
    except sqlite3.OperationalError as exc:
        print(f"FTS5 unavailable ({exc}) - the server will fall back to LIKE")
    finally:
        conn.close()


if __name__ == "__main__":
    # Defaults match the notebook's configuration cell; both paths are
    # relative to the repo root, so run this from there.
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/openalex_raw.jsonl.gz"
    db = sys.argv[2] if len(sys.argv) > 2 else "data/openalex.db"
    create_database(raw, db)
