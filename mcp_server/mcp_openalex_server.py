#!/usr/bin/env python3
"""A from-scratch, MCP 2026-07-28 stdio server over read-only OpenAlex data.

No SDK, no framework: the point is to make the current wire format visible.
The server speaks newline-delimited JSON-RPC 2.0 over stdin/stdout and uses
the stateless protocol introduced in MCP 2026-07-28:

  * ``server/discover`` replaces the initialization handshake
  * every request carries protocol version and capabilities in ``_meta``
  * every result carries ``resultType`` and server identity
  * cacheable discovery/list/resource results include cache hints

Design decisions (each explained in the notebook):
  * read-only enforcement at the connection level, not the prompt level
  * one statement per call (Python's sqlite3 enforces this for us)
  * wall-clock query timeout via SQLite's progress handler
  * explicit, expiring handles for state that spans tool calls
  * all logging to stderr -- stdout belongs exclusively to the protocol

This is intentionally modern-only. Legacy clients that require
``initialize`` (MCP 2025-11-25 and earlier) need a dual-era adapter.
"""

import collections
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data/openalex.db"

PROTOCOL_VERSION = "2026-07-28"
SUPPORTED_PROTOCOL_VERSIONS = [PROTOCOL_VERSION]
SERVER_INFO = {"name": "openalex-sqlite", "version": "0.2.0"}
SERVER_CAPABILITIES = {"tools": {}, "resources": {}}
SERVER_INSTRUCTIONS = (
    "Call list_tables, then describe_table before composing joins. "
    "Use search_works to enter the graph from natural-language concepts."
)
CACHE_TTL_MS = 3_600_000

PAGE_SIZE_DEFAULT = 50
PAGE_SIZE_MAX = 200
QUERY_TIMEOUT_S = 5.0
MAX_OPEN_HANDLES = 32
HANDLE_TTL_S = 300.0
CELL_MAX_CHARS = 400  # truncate huge abstracts in tool output
SQL_VALUE_MAX_BYTES = 1_000_000
SQL_TEXT_MAX_BYTES = 100_000


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
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True,
                           check_same_thread=False)
    conn.execute("PRAGMA query_only = ON;")  # belt on top of braces
    # Output clipping happens after SQLite computes a value. Engine-level
    # limits stop queries such as SELECT randomblob(1_000_000_000) before
    # they can allocate an unbounded cell, and cap oversized SQL text.
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, SQL_VALUE_MAX_BYTES)
    conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, SQL_TEXT_MAX_BYTES)
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
# Result-set handles (explicit state in a stateless protocol)
# --------------------------------------------------------------------------
# A handle is an opaque token the model threads between tool calls. MCP has
# no protocol-level session; from the wire's perspective this is an ordinary
# string result followed by an ordinary string argument. The cursor remains
# application state, so it is bounded by both count and lifetime. A networked,
# multi-instance server would put this state in shared storage or use a signed
# continuation token instead of this process-local dictionary.

open_cursors = collections.OrderedDict()  # handle -> (cursor, expires_at)


def close_handle(handle):
    entry = open_cursors.pop(handle, None)
    if entry is not None:
        entry[0].close()


def reap_expired_handles(now=None):
    now = time.monotonic() if now is None else now
    for handle, (_, expires_at) in list(open_cursors.items()):
        if expires_at <= now:
            close_handle(handle)


def mint_handle(cursor):
    now = time.monotonic()
    reap_expired_handles(now)
    # In this unauthenticated local transport the handle is a bearer token,
    # so keep the full UUID4 entropy rather than exposing a short prefix.
    handle = uuid.uuid4().hex
    while handle in open_cursors:  # astronomically unlikely, still lossless
        handle = uuid.uuid4().hex
    open_cursors[handle] = (cursor, now + HANDLE_TTL_S)
    while len(open_cursors) > MAX_OPEN_HANDLES:      # bound memory:
        old_handle = next(iter(open_cursors))        # evict least recent
        close_handle(old_handle)
    return handle


def cursor_for_handle(handle):
    now = time.monotonic()
    reap_expired_handles(now)
    entry = open_cursors.get(handle)
    if entry is None:
        raise ValueError(f"unknown or expired handle {handle!r}")
    cursor, _ = entry
    open_cursors[handle] = (cursor, now + HANDLE_TTL_S)
    open_cursors.move_to_end(handle)
    return cursor


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
    page_size = args.get("page_size", PAGE_SIZE_DEFAULT)
    # UX guard only -- fast, clear feedback for the model. The actual
    # enforcement is mode=ro + query_only above: even if a clever
    # prompt sneaks past this check, SQLite refuses the write.
    if not sql.lstrip().lower().startswith(("select", "with")):
        raise ValueError("only SELECT / WITH queries are allowed")
    with_deadline(conn)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)  # sqlite3 raises if sql holds >1 statement
        rows = cursor.fetchmany(page_size)
    except sqlite3.Error as exc:
        cursor.close()
        raise ValueError(f"SQL error: {exc}") from exc
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
    page_size = args.get("page_size", PAGE_SIZE_DEFAULT)
    cursor = cursor_for_handle(handle)
    with_deadline(conn)
    rows = cursor.fetchmany(page_size)
    columns, page = rows_to_payload(cursor, rows)
    payload = {"columns": columns, "rows": page,
               "row_count": len(page), "done": len(rows) < page_size}
    if payload["done"]:
        close_handle(handle)
    else:
        payload["handle"] = handle
        payload["next"] = "pass `handle` to fetch_page for more rows"
    return payload


def fts_available(conn):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'works_fts'").fetchone()
    return row is not None


def tool_search_works(conn, args):
    query = args["query"]
    limit = args.get("limit", 10)
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
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "describe_table": {
        "handler": tool_describe_table,
        "description": ("Columns, types, foreign keys and 3 sample rows "
                        "for one table. Use before writing joins."),
        "inputSchema": {
            "type": "object",
            "properties": {"table": {"type": "string",
                                     "minLength": 1,
                                     "description": "table name"}},
            "required": ["table"],
            "additionalProperties": False,
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
                "sql": {"type": "string", "minLength": 1,
                        "description": "a single SELECT or WITH query"},
                "page_size": {"type": "integer", "minimum": 1,
                              "maximum": PAGE_SIZE_MAX,
                              "default": PAGE_SIZE_DEFAULT},
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    "fetch_page": {
        "handler": tool_fetch_page,
        "description": "Fetch the next page of a previous query by handle.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string", "minLength": 1},
                "page_size": {"type": "integer", "minimum": 1,
                              "maximum": PAGE_SIZE_MAX,
                              "default": PAGE_SIZE_DEFAULT},
            },
            "required": ["handle"],
            "additionalProperties": False,
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
                          "minLength": 1,
                          "description": "FTS5 query, e.g. 'chinchilla "
                                         "AND compute'"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50,
                          "default": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def validate_tool_arguments(name, args):
    """Validate the JSON-Schema subset used by this server's tool catalog.

    Advertising an ``inputSchema`` helps clients and models construct calls;
    it does not absolve the server from validating untrusted arguments. The
    implementation stays dependency-free by supporting exactly the keywords
    used above: object properties, required/additionalProperties, primitive
    types, string length, and integer bounds.
    """
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")

    schema = TOOLS[name]["inputSchema"]
    properties = schema.get("properties", {})
    missing = [key for key in schema.get("required", []) if key not in args]
    if missing:
        raise ValueError("missing required argument(s): " + ", ".join(missing))

    if schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(properties))
        if unknown:
            raise ValueError("unknown argument(s): " + ", ".join(unknown))

    for key, value in args.items():
        rule = properties.get(key)
        if rule is None:
            continue
        expected = rule.get("type")
        valid = (
            expected == "string" and isinstance(value, str)
            or expected == "integer"
            and isinstance(value, int) and not isinstance(value, bool)
        )
        if expected and not valid:
            raise ValueError(f"{key!r} must be {expected}")
        if expected == "string" and len(value) < rule.get("minLength", 0):
            raise ValueError(f"{key!r} must not be empty")
        if expected == "integer":
            if "minimum" in rule and value < rule["minimum"]:
                raise ValueError(f"{key!r} must be >= {rule['minimum']}")
            if "maximum" in rule and value > rule["maximum"]:
                raise ValueError(f"{key!r} must be <= {rule['maximum']}")


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

class RpcError(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def send(message):
    # One JSON object per line. json.dumps with no indent never emits a
    # raw newline, so the framing is safe by construction.
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def reply(msg_id, result):
    send({"jsonrpc": "2.0", "id": msg_id, "result": result})


def reply_error(msg_id, code, text, data=None):
    error = {"code": code, "message": text}
    if data is not None:
        error["data"] = data
    send({"jsonrpc": "2.0", "id": msg_id, "error": error})


def complete_result(payload, *, cacheable=False, cache_scope="public"):
    """Wrap an operation payload in the MCP 2026-07-28 result envelope."""
    result = {
        "resultType": "complete",
        **payload,
        "_meta": {"io.modelcontextprotocol/serverInfo": SERVER_INFO},
    }
    if cacheable:
        result["ttlMs"] = CACHE_TTL_MS
        result["cacheScope"] = cache_scope
    return result


def validate_request(msg):
    """Validate modern per-request metadata and return the params object."""
    if msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        raise RpcError(-32600, "invalid JSON-RPC request")

    params = msg.get("params")
    if not isinstance(params, dict):
        raise RpcError(-32602, "params must be an object containing _meta")

    meta = params.get("_meta")
    if not isinstance(meta, dict):
        if msg["method"] == "initialize":
            raise RpcError(
                -32601,
                "initialize is not supported by this MCP 2026-07-28 server",
                {"supported": SUPPORTED_PROTOCOL_VERSIONS},
            )
        raise RpcError(-32602, "missing required params._meta")

    requested = meta.get("io.modelcontextprotocol/protocolVersion")
    if not isinstance(requested, str):
        raise RpcError(
            -32602,
            "missing or invalid "
            "_meta.io.modelcontextprotocol/protocolVersion",
        )
    if requested not in SUPPORTED_PROTOCOL_VERSIONS:
        raise RpcError(
            -32022,
            "Unsupported protocol version",
            {"supported": SUPPORTED_PROTOCOL_VERSIONS,
             "requested": requested},
        )

    capabilities = meta.get("io.modelcontextprotocol/clientCapabilities")
    if not isinstance(capabilities, dict):
        raise RpcError(
            -32602,
            "missing or invalid "
            "_meta.io.modelcontextprotocol/clientCapabilities",
        )
    client_info = meta.get("io.modelcontextprotocol/clientInfo")
    if client_info is not None:
        if not isinstance(client_info, dict):
            raise RpcError(
                -32602,
                "_meta.io.modelcontextprotocol/clientInfo must be an object",
            )
        if not all(
            isinstance(client_info.get(key), str) and client_info[key]
            for key in ("name", "version")
        ):
            raise RpcError(
                -32602,
                "clientInfo requires non-empty string name and version",
            )
    return params


def method_params(params, *, allowed=(), required=()):
    """Return operation parameters after removing protocol-level ``_meta``."""
    values = {key: value for key, value in params.items() if key != "_meta"}
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise RpcError(-32602, "unknown parameter(s): " + ", ".join(unknown))
    missing = [key for key in required if key not in values]
    if missing:
        raise RpcError(-32602,
                       "missing required parameter(s): " + ", ".join(missing))
    return values


def handle_discover(params):
    method_params(params)
    return complete_result({
        "supportedVersions": SUPPORTED_PROTOCOL_VERSIONS,
        "capabilities": SERVER_CAPABILITIES,
        "instructions": SERVER_INSTRUCTIONS,
    }, cacheable=True)


def handle_tools_list(params):
    values = method_params(params, allowed=("cursor",))
    if "cursor" in values:
        raise RpcError(-32602, "invalid or expired tools/list cursor")
    tools = [
        {"name": name, "description": TOOLS[name]["description"],
         "inputSchema": TOOLS[name]["inputSchema"]}
        for name in sorted(TOOLS)
    ]
    return complete_result({"tools": tools}, cacheable=True)


def handle_tools_call(conn, params):
    values = method_params(
        params, allowed=("name", "arguments"), required=("name",))
    name = values["name"]
    if not isinstance(name, str):
        raise RpcError(-32602, "tool name must be a string")
    if name not in TOOLS:
        raise RpcError(-32602, f"unknown tool {name!r}")
    args = values.get("arguments", {})
    if not isinstance(args, dict):
        raise RpcError(-32602, "tool arguments must be an object")
    try:
        validate_tool_arguments(name, args)
        payload = TOOLS[name]["handler"](conn, args)
        return complete_result({
            "content": [{"type": "text",
                         "text": json.dumps(payload, indent=2)}],
            "structuredContent": payload,
            "isError": False,
        })
    except (ValueError, sqlite3.Error) as exc:
        # Expected execution failures are tool results the model can repair.
        # Unexpected programming failures bubble to main as JSON-RPC -32603.
        log(f"tool {name} failed:", exc)
        return complete_result({  # the model should see and self-correct
            "content": [{"type": "text", "text": f"error: {exc}"}],
            "isError": True,
        })


def handle_resources_list(params):
    values = method_params(params, allowed=("cursor",))
    if "cursor" in values:
        raise RpcError(-32602, "invalid or expired resources/list cursor")
    return complete_result({"resources": [{
        "uri": SCHEMA_URI,
        "name": "Database schema",
        "description": "Full DDL of the OpenAlex subset",
        "mimeType": "text/plain",
    }]}, cacheable=True)


def handle_resources_read(conn, params):
    values = method_params(params, allowed=("uri",), required=("uri",))
    uri = values["uri"]
    if not isinstance(uri, str):
        raise RpcError(-32602, "resource URI must be a string")
    if uri != SCHEMA_URI:
        raise RpcError(-32602, f"unknown resource {uri!r}",
                       {"uri": uri})
    return complete_result({"contents": [{
        "uri": SCHEMA_URI,
        "mimeType": "text/plain",
        "text": read_schema(conn),
    }]}, cacheable=True)


def dispatch(conn, method, params):
    if method == "server/discover":
        return handle_discover(params)
    if method == "tools/list":
        return handle_tools_list(params)
    if method == "tools/call":
        return handle_tools_call(conn, params)
    if method == "resources/list":
        return handle_resources_list(params)
    if method == "resources/read":
        return handle_resources_read(conn, params)
    raise RpcError(-32601, f"method not found: {method}")


def handle_notification(msg):
    # Notifications deliberately have no response. This synchronous teaching
    # server cannot interrupt a query already running on the same thread; the
    # 5-second SQLite deadline is its hard backstop.
    if msg.get("method") == "notifications/cancelled":
        params = msg.get("params")
        if not isinstance(params, dict):
            return
        log("cancellation requested for", params.get("requestId"),
            params.get("reason", ""))


def main():
    conn = open_db()
    log("serving", DB_PATH, "over MCP", PROTOCOL_VERSION, "stdio")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                reply_error(None, -32700, "parse error")
                continue

            if not isinstance(msg, dict):
                reply_error(None, -32600, "invalid JSON-RPC request")
                continue
            if "method" not in msg:  # clients cannot send responses in modern
                log("ignoring unexpected response-shaped message")
                continue
            if "id" not in msg:      # notification: no reply allowed
                handle_notification(msg)
                continue

            msg_id = msg["id"]
            if (msg_id is None or isinstance(msg_id, bool)
                    or not isinstance(msg_id, (str, int))):
                reply_error(None, -32600, "request id must be string or integer")
                continue

            try:
                params = validate_request(msg)
                reply(msg_id, dispatch(conn, msg["method"], params))
            except RpcError as exc:
                reply_error(msg_id, exc.code, exc.message, exc.data)
            except Exception as exc:  # last-resort guard: keep loop alive
                log("internal error:", exc)
                reply_error(msg_id, -32603, "internal error")
    finally:
        for handle in list(open_cursors):
            close_handle(handle)
        conn.close()
        log("stopped")


if __name__ == "__main__":
    main()
