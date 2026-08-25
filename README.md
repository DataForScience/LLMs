![GitHub](https://img.shields.io/github/license/DataForScience/LLMs)
[![Twitter @data4sci](https://img.shields.io/twitter/follow/data4sci)](https://twitter.com/intent/follow?screen_name=data4sci)
![GitHub top language](https://img.shields.io/github/languages/top/DataForScience/LLMs)
![GitHub repo size](https://img.shields.io/github/repo-size/DataForScience/LLMs)
![GitHub last commit](https://img.shields.io/github/last-commit/DataForScience/LLMs)

[![Data For Science Substack](https://img.shields.io/badge/Graphs_For_Science-Subscribe-blue)](https://data4sci.substack.com/)
[![Data Science Briefing](https://img.shields.io/badge/Data_Science_Briefing-Subscribe-blue)](https://data4science.ck.page/a63d4cc8d9)

# LLMs for Science

Educational notebooks exploring production-grade agentic systems and LLMs pipelines from first principles. Learn the core patterns behind tools like Claude Code, Cursor's agent mode, and autonomous research assistants by implementing them yourself.

These notebooks teach you to build the scaffolding that transforms an LLM from a text generator into a truly useful everyday tool.


## Learning Path

1. **Start with Notebook 1** (`01 - Basic Agentic Harness.ipynb`) to understand the core concepts
2. **Continue with Notebook 2** (`02 - Small Language Model.ipynb`) to see how language models work from the ground up
3. **Then Notebook 3** (`03 - Advanced Agentic Harness.ipynb`) to upgrade the harness with production-grade patterns
4. **Continue with Notebook 4** (`04 - Evaluating Agentic Harnesses.ipynb`) to measure whether any of it actually works
5. **Finish with Notebook 5** (`05 - MCP Server.ipynb`) to hand real data to an agent through a from-scratch MCP server

Notebooks 3 and 4 share a module, [`d4sci_harness.py`](./d4sci_harness.py) — notebook 3 builds it
step by step, notebook 4 imports it. See [The harness as a module](#the-harness-as-a-module) below.
Notebook 5 likewise keeps its executable pieces in [`mcp_server/`](./mcp_server/) — see
[The MCP server as scripts](#the-mcp-server-as-scripts) below.

## Key Features

- **Runs offline** — Every notebook works end to end on a rule-based mock backend, with no API key
- **Production-ready patterns** — Learn the same techniques used in Claude Code, Cursor, and Devin
- **Hands-on implementation** — Build everything from scratch to understand every design decision
- **Measured, not asserted** — The eval suite in notebook 4 turns "it worked once" into pass rates, cost, and failure-mode distributions


## What You'll Learn

### Notebook 1: Basic Agentic Harness
Build a minimal but complete harness from scratch — the foundation of any autonomous agent system.

**Core concepts:**
- The five components of a harness's core state (goal, trace, memory, budget, status)
- Implementing a control loop that drives an LLM through multi-step tasks
- Defining typed tools the LLM can call safely
- Validating LLM-proposed actions against schemas before execution
- Inspecting execution traces for debugging

**What you'll build:** A single-agent harness that solves multi-step research tasks by repeatedly composing context, asking the LLM what to do next, executing tool calls, and updating state until the goal is met.

### Notebook 2: Small Language Model
Build a complete language model from first principles — counting, not neural networks — to understand what every LLM is really doing under the hood.

**Core concepts:**
- The language modeling task: estimating `P(next word | previous words)`
- Building an n-gram (4-gram) model over the WikiText-103 corpus (~103M words)
- Why text is sparse: word-frequency distributions and **Zipf's law**
- Counting three-word contexts and the single-continuation problem
- **Autoregressive generation** — feeding the output back in to predict the next token
- **Temperature** and sampling — greedy decoding vs. probabilistic sampling

**What you'll build:** A 4-gram model trained on Wikipedia that generates text from a prompt, with a temperature switch that mirrors the same decoding knob exposed by modern LLMs.

### Notebook 3: Advanced Agentic Harness
Upgrade every component toward production-grade systems like Claude Code, Devin, or modern research agents.

**Advanced topics:**
- **Typed tools with Pydantic** — Auto-generated JSON schemas and robust validation
- **DAG orchestration** — Parallel execution of independent tasks instead of sequential processing
- **Multi-tier memory** — Working, episodic, and semantic memory with retrieval
- **Verification hierarchy** — Cheap deterministic checks first, LLM-as-judge only when they pass
- **Multi-agent roles** — Planner / Worker / Critic specialization for robustness
- **Multi-dimensional budgeting** — Graceful degradation under token, time, and cost constraints
- **Error taxonomy** — Transient failures retry in place; missing information escalates to a re-plan
- **Structured tracing** — Full observability and replay capability

**What you'll build:** A three-city comparison agent whose plan contains nine independent fetches — executed in parallel under a concurrency cap — plus a final aggregation step and verifiable output, with trace plots for per-step latency, budget pressure, and tokens by role.

### Notebook 4: Evaluating Agentic Harnesses
One good demo proves the harness *can* work. An eval suite proves it *usually* works — and prices the times it doesn't.

**Core concepts:**
- **Eval suites vs. demos** — the smallest useful loop: run every task, record pass/fail, tokens, cost, and status
- **Adversarial tasks** — one task asks for a city that doesn't exist, to exercise the recovery path rather than the happy path
- **Reading results like a CI dashboard** — failure-mode distribution (`failed_execute` vs. `failed_verify`), regression baselines, and which verification tier produced each verdict
- **Trace-derived plots** — cost per task by outcome, per-role latency against actual wall clock (the gap *is* the parallelism payoff), tokens by role, and budget-pressure trajectories against the degradation threshold
- **Honest instruments** — why the token and cost columns are estimates derived from plan shape, and what that does and doesn't buy you

**Closing three gaps**, each shipped with its own mini-benchmark:
- **Re-planning on failure** — an unknown entity amends the plan instead of aborting the run
- **Real embeddings** — `all-MiniLM-L6-v2` vs. Jaccard retrieval, benchmarked over a labeled query set rather than swapped on faith
- **Specialized workers** — a `FetcherAgent` / `WriterAgent` split routed by capability, because millisecond lookups and multi-second LLM calls do not deserve the same concurrency and retry policy

**What you'll build:** A four-task eval suite run against the harness module, producing a metrics table and four diagnostic plots built entirely from the structured trace the orchestrator already emits — no extra instrumentation.

### Notebook 5: MCP Server
Build a complete MCP server from scratch — raw JSON-RPC over stdio, no SDK — that gives an agent queryable access to a real bibliographic database.

**Core concepts:**
- **Data acquisition done right** — pull a laptop-sized subset of [OpenAlex](https://openalex.org) via cursor pagination and `select=` field trimming, caching the raw JSON separately from the database
- **A normalized SQLite schema** — 9 tables mirroring OpenAlex's entity graph, plus BM25 full-text search via FTS5, because the joins *are* the point
- **EDA as an audit** — know the data's truncations, gaps, and traps (double-counting joins, closed-world citations, right-censored years) *before* the agent finds them
- **The MCP wire format** — newline-delimited JSON-RPC 2.0 over stdio: discovery, per-request `_meta`, result envelopes, cache hints, and error classification, all made visible
- **The latest protocol revision** — the server is native to [MCP `2026-07-28`](https://blog.modelcontextprotocol.io/posts/2026-07-28/), the current spec version, which made the core stateless: `server/discover` replaces the `initialize` handshake, and every request carries its own version and capabilities in `_meta`
- **Defense in depth** — read-only connections at the capability level, server-side argument validation, query deadlines, and expiring pagination handles
- **A from-scratch client** — exercise the server one exchange at a time, including hostile calls and version rejection, so you see every byte on the wire

**What you'll build:** A five-tool, one-resource MCP server over an OpenAlex subset — `list_tables`, `describe_table`, `query` with handle-based pagination, `fetch_page`, and BM25 `search_works` — plus the launch configuration to plug it into a real MCP host.

## The harness as a module

Everything notebook 3 builds step by step — typed tools, the plan DAG, the parallel executor, multi-tier memory, the verification hierarchy, multi-dimensional budgets, structured tracing, and the `Orchestrator` that composes them — also lives in [`d4sci_harness.py`](./d4sci_harness.py) as an importable module. Notebook 3 constructs it; notebook 4 imports it, which is how you would consume it in a real project:

```python
import d4sci_harness as dh
from d4sci_harness import TOOLS, MemoryStore, Orchestrator

llm = dh.set_provider("anthropic")        # or "mock" for offline runs

store = MemoryStore()
orch = Orchestrator(provider=llm, tools=TOOLS, memory=store)

result = await orch.run("Compare Paris and Tokyo.", ["paris", "tokyo"])
print(result.status, result.budget.tokens_used, result.verdict.tier)
```

| Subsystem | Key names |
| :-- | :-- |
| LLM providers | `LLMProvider`, `AnthropicProvider`, `MockProvider`, `set_provider` |
| Typed tools | `TypedTool`, `TOOLS`, `CityArgs`, `AggregateArgs` |
| Plan as a DAG | `PlanDAG`, `PlanNode`, `NodeStatus`, `PlanDAG.plot` |
| Parallel execution | `execute_dag`, `MAX_CONCURRENT`, `MAX_NODE_RETRIES` |
| Memory | `MemoryStore`, `WorkingMemory`, `build_context` |
| Verification | `Verdict`, `ReportCheck`, `deterministic_check_report`, `llm_judge_report`, `verify_report` |
| Agent roles | `PlannerAgent`, `WorkerAgent`, `CriticAgent` |
| Budget + recovery | `BudgetMulti`, `ErrorClass`, `classify_error`, `retry_with_backoff` |
| Tracing | `TraceEvent`, `Tracer` |
| Composition root | `Orchestrator`, `RunResult` |

Two things worth knowing before you extend it:

- **Backends are a one-line swap.** `MockProvider` is fully offline and rule-based; `AnthropicProvider` uses Claude. The code path is identical either way, which is what makes the eval suite runnable in CI.
- **Verification is pluggable.** `Orchestrator.run()` accepts a `check` callable that replaces the deterministic tier wholesale, so you can verify something other than "does this text mention these terms" without touching the escalation logic above it.

## The MCP server as scripts

Notebook 5 explains every design decision in prose, but the executable pieces live in [`mcp_server/`](./mcp_server/) so they can run outside the notebook — from a shell, a Makefile, or an MCP host's launch configuration:

| File | Role |
| :-- | :-- |
| [`mcp_openalex_server.py`](./mcp_server/mcp_openalex_server.py) | The dependency-free MCP stdio server: five tools + one resource over the read-only OpenAlex database |
| [`create_openalex_db.py`](./mcp_server/create_openalex_db.py) | The build pipeline: raw JSONL cache → `parse_work` → two-pass bulk load → FTS index. Run as `python create_openalex_db.py [raw_path] [db_path]` |
| [`mcp_openalex_adapter.py`](./mcp_server/mcp_openalex_adapter.py) | A dual-era adapter that answers the legacy (pre-2026) `initialize` handshake itself and bridges older clients to the stateless server without modifying it |
| [`test_mcp_openalex_server.py`](./mcp_server/test_mcp_openalex_server.py) | Wire-level tests that spawn the server as a subprocess and exercise it exactly as a host would |

Everything on the server's critical path is standard library only (`json`, `sqlite3`, `sys`), so it runs as a bare subprocess wherever a host launches it. The server is modern-only: it requires a host that speaks [MCP `2026-07-28`](https://blog.modelcontextprotocol.io/posts/2026-07-28/), the latest protocol revision, and rejects the legacy `initialize` handshake explicitly — hosts still on an earlier version should launch `mcp_openalex_adapter.py` instead. To register it with a stdio host, point the host at the server script with absolute paths:

```json
{
  "mcpServers": {
    "openalex": {
      "command": "/ABS/PATH/TO/python3",
      "args": ["/ABS/PATH/TO/mcp_server/mcp_openalex_server.py"]
    }
  }
}
```

The server resolves `data/openalex.db` relative to its own location, so it is independent of whatever working directory the host chooses.

## Contents

| Notebooks | Blog post | Content |
| :--: | :--: | :--: |
| **[01 - Basic Agentic Harness.ipynb](./01%20-%20Basic%20Agentic%20Harness.ipynb)** | **[Building a Basic Agentic Harness](https://data4sci.substack.com/p/building-a-basic-agentic-harness)** | Start here to understand the fundamentals |
| **[02 - Small Language Model.ipynb](./02%20-%20Small%20Language%20Model.ipynb)** |  **[Build a (small) language model by counting](https://data4sci.substack.com/p/build-a-small-language-model-by-counting)** | Build an n-gram language model from scratch and generate text |
| **[03 - Advanced Agentic Harness.ipynb](./03%20-%20Advanced%20Agentic%20Harness.ipynb)** | **[Building an Advanced Agentic Harness](https://data4sci.substack.com/p/building-an-advanced-agentic-harness)** | Production-grade patterns: DAG orchestration, memory, verification, and multi-agent roles |
| None | **[Self hosting LLMs with Llama.cpp](https://data4sci.substack.com/p/self-hosting-llms-with-llamacpp)** | Run LLMs on your own hardware: install llama.cpp, serve models over an OpenAI-compatible API, explore GGUF files, and choose the right quantization |
| **[04 - Evaluating Agentic Harnesses.ipynb](./04%20-%20Evaluating%20Agentic%20Harnesses.ipynb)** | *Coming soon* | Eval suites, cost and latency measurement, failure modes, and three measured upgrades |
| **[05 - MCP Server.ipynb](./05%20-%20MCP%20Server.ipynb)** | *Coming soon* | Build an MCP server from scratch: OpenAlex data pipeline, SQLite + FTS5, raw JSON-RPC over stdio, and a from-scratch client |

## Repository Structure

```
LLMs/
├── 01 - Basic Agentic Harness.ipynb        # Notebook 1: Fundamentals
├── 02 - Small Language Model.ipynb         # Notebook 2: n-gram language model
├── 03 - Advanced Agentic Harness.ipynb     # Notebook 3: Production-grade patterns
├── 04 - Evaluating Agentic Harnesses.ipynb # Notebook 4: Eval suite, cost, failure modes
├── 05 - MCP Server.ipynb                   # Notebook 5: MCP server from scratch
├── d4sci_harness.py                        # The harness from notebook 3, importable
├── mcp_server/                              # Notebook 5's executable pieces
│   ├── mcp_openalex_server.py              # The from-scratch MCP stdio server
│   ├── create_openalex_db.py               # OpenAlex → SQLite build pipeline
│   ├── mcp_openalex_adapter.py             # Dual-era protocol adapter
│   └── test_mcp_openalex_server.py         # Wire-level server tests
├── data/                                    # Logos and assets
│   ├── D4Sci_logo_ball.png
│   ├── D4Sci_logo_full.png
│   └── bgoncalves.png
├── d4sci.mplstyle                           # Custom matplotlib style
├── pyproject.toml                           # Dependency manifest (for `uv sync`)
├── uv.lock                                  # Lock file for reproducible builds
└── LICENSE                                  # MIT License
```

## Setup

### 1) Install dependencies (recommended: `uv`)

1. Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create an environment and install dependencies:

```bash
git clone https://github.com/DataForScience/LLMs.git
cd LLMs
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### 2) API Keys (Optional)

The notebooks run end-to-end in **mock mode** without any API keys. They include rule-based mock LLM providers that are smart enough to drive the demos.

Notebooks 3 and 4 ship with `BACKEND = "anthropic"`, so to run them offline set:

```python
BACKEND = "mock"
```

To use the real API instead, export your Anthropic API key before launching Jupyter:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Notebook 5 needs an `OPENALEX_API_KEY` only for a fresh data fetch — the cached raw
data lets the rest of the notebook run without credentials:

```bash
export OPENALEX_API_KEY=...
```

Mock mode is deterministic, which makes it the right choice for CI and for following along.
Real-backend runs are not — plans vary between runs, so expect the eval suite's pass rate
to move around. That variability is the point of measuring it.

### 3) Launch notebooks

```bash
jupyter notebook
```

---

## Questions?

Reach out at <a href="mailto:info@data4sci.com">info@data4sci.com</a> or open an issue if something isn't working.

## Author

<table border="0">
 <tr>
	<td>
	  <img src="data/bgoncalves.png" alt="Bruno Gonçalves" width="150" height="150" style="border-radius: 50%; object-fit: cover;">
	</td>
	<td>
	  <h2>Bruno Gonçalves</h2>
	  <h3>Data For Science, Inc.</h3>
	  <p>
			Web: <a href="http://www.data4sci.com/">www.data4sci.com</a><br>
			Twitter/X: <a href="https://twitter.com/bgoncalves">@bgoncalves</a><br>
			LinkedIn: <a href="https://www.linkedin.com/in/bmtgoncalves/">@bmtgoncalves</a><br>
			Email: <a href="mailto:info@data4sci.com">info@data4sci.com</a><br>
			Schedule a Call: <a href="https://data4sci.com/call">data4sci.com/call</a>
	  </p>
	</td>
 </tr>
</table>
