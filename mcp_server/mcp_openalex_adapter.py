#!/usr/bin/env python3
"""Dual-era adapter: standard MCP (``initialize``) <-> MCP 2026-07-28.

``mcp_openalex_server.py`` intentionally speaks only the fictional, stateless
"MCP 2026-07-28" wire format: it rejects ``initialize`` and instead expects
``server/discover`` plus per-request ``_meta`` carrying the protocol version
and client capabilities. Real clients (pi included) still speak the released
protocol that begins with an ``initialize`` handshake.

This adapter bridges the two eras WITHOUT modifying the server:

  * It is the stdio program the client launches.
  * It answers ``initialize`` / ``ping`` / ``notifications/initialized``
    itself, using the released protocol.
  * For every real operation (tools/resources) it spawns the unchanged
    server as a subprocess, injects the ``_meta`` block the server demands,
    forwards the request, then unwraps the 2026-07-28 result envelope back
    into a plain released-protocol result.

Everything the server logs goes to *its* stderr, which we inherit, so it
still lands in the same place. Our own diagnostics use a distinct prefix.
"""

import json
import subprocess
import sys
from pathlib import Path

SERVER_PATH = Path(__file__).resolve().parent / "mcp_openalex_server.py"

# The protocol version the downstream server insists on.
DOWNSTREAM_PROTOCOL = "2026-07-28"
# What we advertise upstream if the client omits a version.
DEFAULT_UPSTREAM_PROTOCOL = "2025-06-18"

ADAPTER_INFO = {"name": "openalex-dual-era-adapter", "version": "1.0.0"}
FALLBACK_SERVER_INFO = {"name": "openalex-sqlite", "version": "0.2.0"}

# Keys that belong to the 2026-07-28 result envelope, not the operation
# payload the released protocol expects.
ENVELOPE_KEYS = {"resultType", "_meta", "ttlMs", "cacheScope"}

# Methods that map straight through to the downstream server (with _meta
# injected and the result envelope stripped on the way back).
PASS_THROUGH = {
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/read",
    "resources/templates/list",
    "prompts/list",
    "prompts/get",
}


def log(*args):
    print("[openalex-adapter]", *args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Upstream (client) transport
# --------------------------------------------------------------------------

def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def reply(msg_id, result):
    send({"jsonrpc": "2.0", "id": msg_id, "result": result})


def reply_error(msg_id, code, text, data=None):
    error = {"code": code, "message": text}
    if data is not None:
        error["data"] = data
    send({"jsonrpc": "2.0", "id": msg_id, "error": error})


# --------------------------------------------------------------------------
# Downstream (server) transport
# --------------------------------------------------------------------------

class Downstream:
    """A single long-lived subprocess speaking MCP 2026-07-28 over stdio."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit: server logs land where they always did
            text=True,
            bufsize=1,
        )
        self._id = 0

    def _next_id(self):
        self._id += 1
        return f"adapter-{self._id}"

    @staticmethod
    def _with_meta(params, client_info):
        meta = dict((params or {}).get("_meta") or {})
        meta["io.modelcontextprotocol/protocolVersion"] = DOWNSTREAM_PROTOCOL
        meta.setdefault("io.modelcontextprotocol/clientCapabilities", {})
        if client_info:
            meta["io.modelcontextprotocol/clientInfo"] = client_info
        else:
            meta.setdefault(
                "io.modelcontextprotocol/clientInfo", ADAPTER_INFO)
        merged = {k: v for k, v in (params or {}).items() if k != "_meta"}
        merged["_meta"] = meta
        return merged

    def request(self, method, params, client_info):
        """Send one request, block for its matching response."""
        if self.proc.poll() is not None:
            raise RuntimeError("downstream server has exited")
        req_id = self._next_id()
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": self._with_meta(params, client_info),
        }
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("downstream server closed the connection")
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                log("dropping non-JSON line from server:", line[:200])
                continue
            if resp.get("id") == req_id:
                return resp
            # The server only speaks in response to us and never emits
            # unsolicited traffic, but stay robust just in case.
            log("ignoring out-of-band message id:", resp.get("id"))

    def notify(self, method, params):
        if self.proc.poll() is not None:
            return
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def close(self):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def unwrap(result):
    """Strip the 2026-07-28 envelope, leaving the released-protocol payload."""
    return {k: v for k, v in result.items() if k not in ENVELOPE_KEYS}


# --------------------------------------------------------------------------
# Handshake handled locally, in the released protocol
# --------------------------------------------------------------------------

def handle_initialize(down, msg_id, params, client_info):
    requested = (params or {}).get("protocolVersion")
    upstream_version = (
        requested if isinstance(requested, str) and requested
        else DEFAULT_UPSTREAM_PROTOCOL)

    resp = down.request("server/discover", {}, client_info)
    if "error" in resp:
        reply_error(msg_id, resp["error"].get("code", -32603),
                    "downstream discover failed: "
                    + resp["error"].get("message", ""),
                    resp["error"].get("data"))
        return

    result = resp.get("result", {})
    server_info = (result.get("_meta", {})
                   .get("io.modelcontextprotocol/serverInfo",
                        FALLBACK_SERVER_INFO))
    reply(msg_id, {
        "protocolVersion": upstream_version,
        "capabilities": result.get("capabilities", {}),
        "serverInfo": server_info,
        "instructions": result.get("instructions", ""),
    })


def handle_pass_through(down, msg_id, method, params, client_info):
    resp = down.request(method, params, client_info)
    if "error" in resp:
        err = resp["error"]
        reply_error(msg_id, err.get("code", -32603),
                    err.get("message", "downstream error"), err.get("data"))
        return
    reply(msg_id, unwrap(resp.get("result", {})))


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    down = Downstream()
    log("bridging released MCP <->", DOWNSTREAM_PROTOCOL, "via", SERVER_PATH)
    client_info = None
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
            if not isinstance(msg, dict) or "method" not in msg:
                continue

            method = msg["method"]

            # Notifications (no id): never get a reply.
            if "id" not in msg:
                if method == "notifications/initialized":
                    continue  # handshake artifact; server has no concept of it
                if method == "notifications/cancelled":
                    down.notify(method, msg.get("params"))
                continue

            msg_id = msg["id"]
            params = msg.get("params") or {}

            try:
                if method == "initialize":
                    ci = params.get("clientInfo")
                    if isinstance(ci, dict):
                        client_info = {
                            "name": str(ci.get("name", "unknown")) or "unknown",
                            "version": str(ci.get("version", "0")) or "0",
                        }
                    handle_initialize(down, msg_id, params, client_info)
                elif method == "ping":
                    reply(msg_id, {})
                elif method in PASS_THROUGH:
                    handle_pass_through(down, msg_id, method, params,
                                        client_info)
                else:
                    reply_error(msg_id, -32601,
                                f"method not found: {method}")
            except Exception as exc:  # keep the bridge alive
                log("adapter error on", method, ":", exc)
                reply_error(msg_id, -32603, f"adapter error: {exc}")
    finally:
        down.close()
        log("stopped")


if __name__ == "__main__":
    main()
