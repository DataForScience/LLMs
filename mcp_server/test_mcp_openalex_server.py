"""Wire-level tests for the dependency-free MCP server."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SERVER = Path(__file__).with_name("mcp_openalex_server.py").resolve()
PROTOCOL_VERSION = "2026-07-28"


class StdioClient:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            cwd=self.tmp.name,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.next_id = 0

    @staticmethod
    def meta(version=PROTOCOL_VERSION):
        return {
            "io.modelcontextprotocol/protocolVersion": version,
            "io.modelcontextprotocol/clientInfo": {
                "name": "unittest-client",
                "version": "1.0",
            },
            "io.modelcontextprotocol/clientCapabilities": {},
        }

    def request(self, method, params=None, *, version=PROTOCOL_VERSION,
                add_meta=True, meta=None):
        self.next_id += 1
        params = dict(params or {})
        if add_meta:
            params["_meta"] = meta if meta is not None else self.meta(version)
        message = {
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": method,
            "params": params,
        }
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(self.proc.stderr.read())
        return json.loads(line)

    def notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        stderr = self.proc.stderr.read()
        if not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc.stderr.close()
        self.tmp.cleanup()
        return stderr


class ModernMcpServerTests(unittest.TestCase):
    def setUp(self):
        self.client = StdioClient()

    def tearDown(self):
        self.client.close()

    def assert_complete(self, response):
        self.assertEqual(response["result"]["resultType"], "complete")
        self.assertEqual(
            response["result"]["_meta"][
                "io.modelcontextprotocol/serverInfo"]["name"],
            "openalex-sqlite",
        )

    def test_discovery_is_modern_and_cacheable(self):
        response = self.client.request("server/discover")

        self.assert_complete(response)
        result = response["result"]
        self.assertEqual(result["supportedVersions"], [PROTOCOL_VERSION])
        self.assertEqual(result["capabilities"],
                         {"tools": {}, "resources": {}})
        self.assertGreater(result["ttlMs"], 0)
        self.assertEqual(result["cacheScope"], "public")

    def test_every_request_validates_protocol_metadata(self):
        missing = self.client.request(
            "tools/list", add_meta=False)
        self.assertEqual(missing["error"]["code"], -32602)

        incomplete_meta = self.client.meta()
        incomplete_meta.pop("io.modelcontextprotocol/protocolVersion")
        incomplete = self.client.request(
            "tools/list", meta=incomplete_meta)
        self.assertEqual(incomplete["error"]["code"], -32602)

        legacy = self.client.request(
            "tools/list", version="2025-11-25")
        self.assertEqual(legacy["error"]["code"], -32022)
        self.assertEqual(
            legacy["error"]["data"]["supported"], [PROTOCOL_VERSION])

        initialize = self.client.request(
            "initialize",
            {"protocolVersion": "2025-11-25", "capabilities": {}},
            add_meta=False,
        )
        self.assertEqual(initialize["error"]["code"], -32601)
        self.assertIn("2026-07-28", initialize["error"]["message"])

    def test_malformed_notification_is_ignored_without_crashing(self):
        self.client.notify("notifications/cancelled", params=[])

        response = self.client.request("server/discover")
        self.assert_complete(response)

    def test_tools_are_deterministic_and_arguments_are_validated(self):
        listed = self.client.request("tools/list")
        self.assert_complete(listed)
        names = [tool["name"] for tool in listed["result"]["tools"]]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), 5)

        invalid = self.client.request("tools/call", {
            "name": "query",
            "arguments": {"sql": "SELECT 1", "page_size": 0},
        })
        self.assert_complete(invalid)
        self.assertTrue(invalid["result"]["isError"])
        self.assertIn("must be >= 1",
                      invalid["result"]["content"][0]["text"])

        unknown = self.client.request(
            "tools/call", {"name": "not_a_tool", "arguments": {}})
        self.assertEqual(unknown["error"]["code"], -32602)

    def test_query_paginates_with_explicit_expiring_handle(self):
        first = self.client.request("tools/call", {
            "name": "query",
            "arguments": {
                "sql": "SELECT id FROM works ORDER BY id",
                "page_size": 1,
            },
        })
        self.assert_complete(first)
        payload = first["result"]["structuredContent"]
        self.assertFalse(payload["done"])
        self.assertEqual(len(payload["handle"]), 32)

        second = self.client.request("tools/call", {
            "name": "fetch_page",
            "arguments": {"handle": payload["handle"], "page_size": 1},
        })
        self.assert_complete(second)
        self.assertEqual(second["result"]["structuredContent"]["row_count"], 1)

    def test_read_only_capability_blocks_prefix_bypass(self):
        blocked = self.client.request("tools/call", {
            "name": "query",
            "arguments": {
                "sql": (
                    "WITH x AS (SELECT 1) "
                    "INSERT INTO works (id) SELECT 'W0' FROM x"
                ),
            },
        })
        self.assert_complete(blocked)
        self.assertTrue(blocked["result"]["isError"])
        self.assertIn(
            "attempt to write a readonly database",
            blocked["result"]["content"][0]["text"],
        )

    def test_sqlite_value_limit_blocks_large_allocations(self):
        blocked = self.client.request("tools/call", {
            "name": "query",
            "arguments": {"sql": "SELECT randomblob(2000000)"},
        })
        self.assert_complete(blocked)
        self.assertTrue(blocked["result"]["isError"])
        self.assertIn(
            "string or blob too big",
            blocked["result"]["content"][0]["text"],
        )

    def test_schema_resource_is_cacheable_and_validates_uri(self):
        response = self.client.request(
            "resources/read", {"uri": "openalex://schema"})
        self.assert_complete(response)
        self.assertTrue(
            response["result"]["contents"][0]["text"].startswith("CREATE"))
        self.assertEqual(response["result"]["cacheScope"], "public")

        missing = self.client.request(
            "resources/read", {"uri": "openalex://missing"})
        self.assertEqual(missing["error"]["code"], -32602)


if __name__ == "__main__":
    unittest.main()
