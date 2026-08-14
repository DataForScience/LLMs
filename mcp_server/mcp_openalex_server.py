#!/usr/bin/env python3
"""A from-scratch MCP server (stdio transport) exposing a read-only
OpenAlex SQLite subset.

No SDK, no framework: the point is to see the wire format. The server
speaks JSON-RPC 2.0, one message per line, over stdin/stdout.

Design decisions (each explained in the notebook):
  * read-only enforcement at the connection level, not the prompt level
  * one statement per call (Python's sqlite3 enforces this for us)
  * wall-clock query timeout via SQLite's progress handler
  * pagination through server-minted handles passed as tool arguments
    (the pattern the 2026-07-28 stateless spec recommends)
  * ALL logging to stderr -- stdout belongs to the protocol
"""

import collections
import json
import sqlite3
import sys
import time
import uuid

DB_PATH = "data/openalex.db"

SERVER_INFO = {"name": "openalex-sqlite", "version": "0.1.0"}
SUPPORTED_PROTOCOL_VERSIONS = {
    "2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25",
}
LATEST_PROTOCOL_VERSION = "2025-06-18"

PAGE_SIZE_DEFAULT = 50
PAGE_SIZE_MAX = 200
QUERY_TIMEOUT_S = 5.0
MAX_OPEN_HANDLES = 32
CELL_MAX_CHARS = 400  # truncate huge abstracts in tool output


def log(*args):
    """stderr only. stdout is the transport: one stray print() there and
    the client's JSON parser dies. This is the classic stdio-MCP bug."""
    print("[openalex-mcp]", *args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Database access
# --------------------------------------------------------------------------

def open_db(path=DB_PATH):
    # mode=ro: SQLite itself refuses writes -- a *capability* restriction.
    # The model never gets a connection that could write, so prompt
    # injection cannot escalate into data modification.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                           check_same_thread=False)
    conn.execute("PRAGMA query_only = ON;")  # belt on top of braces
    return conn


def with_deadline(conn, seconds=QUERY_TIMEOUT_S):
    """Abort any statement that runs longer than `seconds`.

    The progress handler is called every N SQLite VM ops; returning a
    non-zero value aborts the statement with OperationalError. This is
    the only reliable way to bound runaway queries (think accidental
    cross joins) inside a single-threaded server.
    """
    deadline = time.monotonic() + seconds
    conn.set_progress_handler(
        lambda: 1 if time.monotonic() > deadline else 0, 50_000)


def clip(value):
    if isinstance(value, str) and len(value) > CELL_MAX_CHARS:
        return value[:CELL_MAX_CHARS] + "…"
    return value


def rows_to_payload(cursor, rows):
    columns = [d[0] for d in cursor.description] if cursor.description else []
    return columns, [[clip(v) for v in row] for row in rows]


# --------------------------------------------------------------------------
# Result-set handles (pagination without sessions)
# --------------------------------------------------------------------------
# A handle is an opaque token the *model* threads between tool calls.
# State lives in an explicit, inspectable place -- a dict keyed by the
# token -- not in the transport. This is exactly the migration path the
# 2026-07-28 spec prescribes now that protocol-level sessions are gone.

open_cursors = collections.OrderedDict()  # handle -> live cursor


def mint_handle(cursor):
    handle = uuid.uuid4().hex[:12]
    open_cursors[handle] = cursor
    while len(open_cursors) > MAX_OPEN_HANDLES:      # bound memory:
        _, old = open_cursors.popitem(last=False)    # evict oldest
        old.close()
    return handle


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def tool_list_tables(conn, args):
    with_deadline(conn)
    cur = conn.execute(
        """SELECT name FROM sqlite_master
           WHERE type IN ('table', 'view')
             AND name NOT LIKE 'sqlite_%'
             AND name NOT GLOB 'works_fts_*'   -- hide FTS shadow tables
           ORDER BY name""")
    tables = [r[0] for r in cur.fetchall()]
    counts = {}
    for t in tables:
        counts[t] = conn.execute(
            f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    return {"tables": [{"name": t, "rows": counts[t]} for t in tables]}


def tool_describe_table(conn, args):
    table = args["table"]
    known = {t["name"] for t in tool_list_tables(conn, {})["tables"]}
    if table not in known:
        raise ValueError(f"unknown table {table!r}; call list_tables first")
    # Identifier validated against the catalog above, so interpolation
    # here is safe -- placeholders cannot bind identifiers in SQL.
    with_deadline(conn)
    cols = [{"name": r[1], "type": r[2], "pk": bool(r[5])}
            for r in conn.execute(f'PRAGMA table_info("{table}")')]
    fks = [{"column": r[3], "references": f"{r[2]}({r[4]})"}
           for r in conn.execute(f'PRAGMA foreign_key_list("{table}")')]
    cur = conn.execute(f'SELECT * FROM "{table}" LIMIT 3')
    columns, sample = rows_to_payload(cur, cur.fetchall())
    return {"table": table, "columns": cols, "foreign_keys": fks,
            "sample_rows": {"columns": columns, "rows": sample}}


def tool_query(conn, args):
    sql = args["sql"]
    page_size = min(int(args.get("page_size", PAGE_SIZE_DEFAULT)),
                    PAGE_SIZE_MAX)
    # UX guard only -- fast, clear feedback for the model. The actual
    # enforcement is mode=ro + query_only above: even if a clever
    # prompt sneaks past this check, SQLite refuses the write.
    if not sql.lstrip().lower().startswith(("select", "with")):
        raise ValueError("only SELECT / WITH queries are allowed")
    with_deadline(conn)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)  # sqlite3 raises if sql holds >1 statement
    except sqlite3.OperationalError as exc:
        raise ValueError(f"SQL error: {exc}") from exc
    rows = cursor.fetchmany(page_size)
    columns, page = rows_to_payload(cursor, rows)
    payload = {"columns": columns, "rows": page,
               "row_count": len(page), "done": True}
    if len(rows) == page_size:            # maybe more -- mint a handle
        payload["done"] = False
        payload["handle"] = mint_handle(cursor)
        payload["next"] = "pass `handle` to fetch_page for more rows"
    else:
        cursor.close()
    return payload


def tool_fetch_page(conn, args):
    handle = args["handle"]
    page_size = min(int(args.get("page_size", PAGE_SIZE_DEFAULT)),
                    PAGE_SIZE_MAX)
    cursor = open_cursors.get(handle)
    if cursor is None:
        raise ValueError(f"unknown or expired handle {handle!r}")
    with_deadline(conn)
    rows = cursor.fetchmany(page_size)
    columns, page = rows_to_payload(cursor, rows)
    payload = {"columns": columns, "rows": page,
               "row_count": len(page), "handle": handle,
               "done": len(rows) < page_size}
    if payload["done"]:
        open_cursors.pop(handle).close()
    return payload


def fts_available(conn):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'works_fts'").fetchone()
    return row is not None


def tool_search_works(conn, args):
    query = args["query"]
    limit = min(int(args.get("limit", 10)), 50)
    with_deadline(conn)
    if fts_available(conn):
        # bm25(): smaller = more relevant, per SQLite convention.
        cur = conn.execute(
            """SELECT f.work_id, w.title, w.publication_year,
                      w.cited_by_count, round(bm25(works_fts), 2) AS rank
               FROM works_fts f JOIN works w ON w.id = f.work_id
               WHERE works_fts MATCH ?
               ORDER BY rank LIMIT ?""", (query, limit))
    else:  # graceful degradation if this SQLite lacks FTS5
        cur = conn.execute(
            """SELECT id, title, publication_year, cited_by_count,
                      NULL AS rank
               FROM works WHERE title LIKE '%' || ? || '%'
               ORDER BY cited_by_count DESC LIMIT ?""", (query, limit))
    columns, page = rows_to_payload(cur, cur.fetchall())
    return {"columns": columns, "rows": page, "row_count": len(page)}


TOOLS = {
    "list_tables": {
        "handler": tool_list_tables,
        "description": ("List every table in the OpenAlex subset with its "
                        "row count. Call this first to orient yourself."),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "describe_table": {
        "handler": tool_describe_table,
        "description": ("Columns, types, foreign keys and 3 sample rows "
                        "for one table. Use before writing joins."),
        "inputSchema": {
            "type": "object",
            "properties": {"table": {"type": "string",
                                     "description": "table name"}},
            "required": ["table"],
        },
    },
    "query": {
        "handler": tool_query,
        "description": ("Run one read-only SQL statement (SELECT or WITH). "
                        "Returns the first page of rows plus a `handle` "
                        "when more are available."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "a single SELECT"},
                "page_size": {"type": "integer", "minimum": 1,
                              "maximum": PAGE_SIZE_MAX,
                              "default": PAGE_SIZE_DEFAULT},
            },
            "required": ["sql"],
        },
    },
    "fetch_page": {
        "handler": tool_fetch_page,
        "description": "Fetch the next page of a previous query by handle.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "page_size": {"type": "integer", "minimum": 1,
                              "maximum": PAGE_SIZE_MAX},
            },
            "required": ["handle"],
        },
    },
    "search_works": {
        "handler": tool_search_works,
        "description": ("Full-text search over titles and abstracts "
                        "(BM25-ranked). Better than guessing LIKE "
                        "patterns with `query`."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "FTS5 query, e.g. 'chinchilla "
                                         "AND compute'"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50,
                          "default": 10},
            },
            "required": ["query"],
        },
    },
}


# --------------------------------------------------------------------------
# Resources: app-controlled context, as opposed to model-invoked tools.
# A host can inject the schema into the prompt *before* the model acts,
# saving it a tool round-trip.
# --------------------------------------------------------------------------

SCHEMA_URI = "openalex://schema"


def read_schema(conn):
    ddl = conn.execute(
        """SELECT sql FROM sqlite_master
           WHERE sql IS NOT NULL AND name NOT GLOB 'works_fts_*'
           ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name""")
    return ";\n\n".join(r[0] for r in ddl.fetchall()) + ";"


# --------------------------------------------------------------------------
# JSON-RPC plumbing
# --------------------------------------------------------------------------

def send(message):
    # One JSON object per line. json.dumps with no indent never emits a
    # raw newline, so the framing is safe by construction.
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def reply(msg_id, result):
    send({"jsonrpc": "2.0", "id": msg_id, "result": result})


def reply_error(msg_id, code, text):
    send({"jsonrpc": "2.0", "id": msg_id,
          "error": {"code": code, "message": text}})


def handle_initialize(params):
    requested = params.get("protocolVersion", "")
    # Version negotiation: echo the client's version when we support it,
    # otherwise answer with the newest one we do -- the client then
    # decides whether it can live with that.
    version = (requested if requested in SUPPORTED_PROTOCOL_VERSIONS
               else LATEST_PROTOCOL_VERSION)
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {}, "resources": {}},
        "serverInfo": SERVER_INFO,
    }


def handle_tools_call(conn, params):
    name = params.get("name")
    if name not in TOOLS:
        raise ValueError(f"unknown tool {name!r}")
    args = params.get("arguments") or {}
    try:
        payload = TOOLS[name]["handler"](conn, args)
        return {
            "content": [{"type": "text",
                         "text": json.dumps(payload, indent=2)}],
            "structuredContent": payload,
            "isError": False,
        }
    except Exception as exc:  # tool errors are *results*, not RPC errors:
        log(f"tool {name} failed:", exc)
        return {              # the model should see them and self-correct
            "content": [{"type": "text", "text": f"error: {exc}"}],
            "isError": True,
        }


def main():
    conn = open_db()
    log("serving", DB_PATH, "over stdio")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply_error(None, -32700, "parse error")
            continue

        method, msg_id = msg.get("method"), msg.get("id")

        if method is None:        # a response to a server request --
            continue              # we never send any, so ignore

        if msg_id is None:        # notification: no reply allowed
            if method == "notifications/initialized":
                log("client ready")
            continue

        try:
            if method == "initialize":
                reply(msg_id, handle_initialize(msg.get("params") or {}))
            elif method == "ping":
                reply(msg_id, {})
            elif method == "tools/list":
                reply(msg_id, {"tools": [
                    {"name": n, "description": t["description"],
                     "inputSchema": t["inputSchema"]}
                    for n, t in TOOLS.items()]})
            elif method == "tools/call":
                reply(msg_id, handle_tools_call(conn, msg.get("params") or {}))
            elif method == "resources/list":
                reply(msg_id, {"resources": [{
                    "uri": SCHEMA_URI, "name": "Database schema",
                    "description": "Full DDL of the OpenAlex subset",
                    "mimeType": "text/plain"}]})
            elif method == "resources/read":
                uri = (msg.get("params") or {}).get("uri")
                if uri != SCHEMA_URI:
                    reply_error(msg_id, -32602, f"unknown resource {uri!r}")
                else:
                    reply(msg_id, {"contents": [{
                        "uri": SCHEMA_URI, "mimeType": "text/plain",
                        "text": read_schema(conn)}]})
            else:
                reply_error(msg_id, -32601, f"method not found: {method}")
        except Exception as exc:   # last-resort guard: never crash the loop
            log("internal error:", exc)
            reply_error(msg_id, -32603, f"internal error: {exc}")


if __name__ == "__main__":
    main()
