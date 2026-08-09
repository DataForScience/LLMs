"""d4sci_harness — Advanced agentic harness primitives.

Factored out of ``03 - Advanced Agentic Harness.ipynb`` so that follow-up
notebooks (``04 - Evaluating Agentic Harnesses.ipynb``) can import the harness
instead of redefining it. The module contains the core primitives built in
Sections 1-10 of notebook 03:

- Pluggable LLM providers (``LLMProvider``, ``AnthropicProvider``, ``MockProvider``)
- Typed tools with Pydantic schemas (``TypedTool`` and the ``TOOLS`` registry)
- The plan as a DAG (``PlanDAG``, ``PlanNode``, ``NodeStatus``)
- Parallel execution with asyncio (``execute_dag``)
- Multi-tier memory (``MemoryStore``, ``WorkingMemory``, ``build_context``)
- Verification hierarchy (``verify_report`` and friends)
- Multi-agent roles (``PlannerAgent``, ``WorkerAgent``, ``CriticAgent``)
- Budgeting and recovery (``BudgetMulti``, ``classify_error``, ``retry_with_backoff``)
- Structured tracing (``TraceEvent``, ``Tracer``)
- The composition root (``Orchestrator``, ``RunResult``)

The LLM-backed tools (``summarize_city``, ``aggregate_report``) call the
module-level ``llm`` provider. It defaults to the offline ``MockProvider``;
switch backends with ``set_provider("anthropic")`` or by passing a provider
instance.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import json
import asyncio
import time
import os
import re
import random

import numpy as np
from pydantic import BaseModel, Field, ValidationError

# Optional dependencies: the harness degrades gracefully without them.
try:
    import anthropic
except ImportError:                     # pragma: no cover
    anthropic = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:                     # pragma: no cover
    SentenceTransformer = None

# Toggle semantic retrieval: True = MiniLM embeddings; False = Jaccard-only (lighter/offline)
USE_REAL_EMBEDDINGS = SentenceTransformer is not None


# =============================================================================
# 1. Pluggable LLM provider
# =============================================================================

class LLMProvider:
    """Shared interface. Subclass to plug in a different backend."""

    def complete(self, system: str, user: str, role: str = "default") -> str:
        raise NotImplementedError

    async def acomplete(self, system: str, user: str, role: str = "default") -> str:
        # Wrap sync call in a thread; works for any SDK.
        return await asyncio.to_thread(self.complete, system, user, role)


def _extract_json_text(text: str) -> str:
    """Best-effort extraction of a JSON payload from an LLM response.

    Handles: raw JSON, ```json ... ``` / ``` ... ``` fences, and prose
    surrounding a JSON object/array. Returns '' if nothing JSON-like is found.
    """
    if not text:
        return ""
    s = text.strip()
    # Strip markdown code fences if present
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    # If it already looks like JSON, return as-is
    if s.startswith("{") or s.startswith("["):
        return s
    # Otherwise pull out the first {...} or [...] block
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    return m.group(1).strip() if m else s


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        if anthropic is None:
            raise RuntimeError("Install the `anthropic` package to use AnthropicProvider.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY in your environment.")
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, user: str, role: str = "default") -> str:
        # Nudge the model toward strict JSON for roles that are parsed downstream.
        structured_roles = {"planner", "replanner", "summarizer", "aggregator", "critic"}
        sys_prompt = system
        if role in structured_roles:
            sys_prompt = (
                system
                + "\n\nRespond with ONLY a single JSON value. No prose, "
                  "no markdown fences, no commentary."
            )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=sys_prompt,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
        return _extract_json_text(raw) if role in structured_roles else raw


# --- Mock knowledge, shared by tools ---
CITY_FACTS = {
    "paris":    {"population": 2_100_000,  "timezone": "CET",  "country": "France"},
    "tokyo":    {"population": 13_960_000, "timezone": "JST",  "country": "Japan"},
    "new york": {"population": 8_335_000,  "timezone": "EST",  "country": "USA"},
    "london":   {"population": 8_982_000,  "timezone": "GMT",  "country": "UK"},
    "sydney":   {"population": 5_312_000,  "timezone": "AEDT", "country": "Australia"},
}


class MockProvider(LLMProvider):
    def complete(self, system: str, user: str, role: str = "default") -> str:
        # Planner: produce a DAG decomposition
        if role == "planner":
            cities = self._extract_cities(user)
            return json.dumps(self._build_plan(cities))

        # Re-planner: handle failures and produce amended DAG
        if role == "replanner":
            cities = self._extract_cities(user)
            # Build plan only for known cities
            known_cities = [c for c in cities if c in CITY_FACTS]
            if known_cities:
                return json.dumps(self._build_plan(known_cities))
            else:
                # No valid cities, return minimal plan
                return json.dumps({"nodes": []})

        # Summarizer worker: produce a one-line city summary
        if role == "summarizer":
            city = self._extract_city(user)
            facts = CITY_FACTS.get(city.lower(), {})
            country = facts.get("country", "its country")
            pop = facts.get("population", 0)
            return json.dumps({
                "summary": (
                    f"{city.title()} is a major city in {country} "
                    f"with roughly {pop/1e6:.1f}M residents."
                )
            })

        # Aggregator: compose a markdown report from the gathered facts
        if role == "aggregator":
            return json.dumps({
                "report": self._compose_report(user),
            })

        # Critic: judge the report against acceptance criteria.
        # The user string contains "GOAL:\n...\n\nREPORT:\n...". We extract the
        # report portion and check it mentions multiple cities and has numeric
        # evidence (e.g., population figures).
        if role == "critic":
            report_part = user.split("REPORT:", 1)[-1].lower()
            cities_mentioned = sum(
                1 for c in CITY_FACTS if c in report_part
            )
            has_numbers = bool(re.search(r"\d{3,}", report_part))
            passes = cities_mentioned >= 2 and has_numbers
            return json.dumps({
                "pass": passes,
                "reason": (
                    f"Report covers {cities_mentioned} cities with data."
                    if passes
                    else "Report is missing cities or quantitative data."
                ),
            })

        return json.dumps({"error": f"unknown role {role!r}"})

    # --- helpers below are mock-only; a real LLM wouldn't need them ---
    def _extract_cities(self, text: str) -> list[str]:
        found = [c for c in CITY_FACTS if c in text.lower()]
        # Also check for unknown cities mentioned in the text
        for word in ["atlantis", "unknown", "narnia"]:
            if word in text.lower():
                found.append(word)
        return found or ["paris", "tokyo", "new york"]

    def _extract_city(self, text: str) -> str:
        for c in CITY_FACTS:
            if c in text.lower():
                return c
        return "paris"

    def _build_plan(self, cities: list[str]) -> dict:
        nodes = []
        fetch_ids_by_city = {}
        for city in cities:
            pop_id = f"pop_{city.replace(' ', '_')}"
            tz_id = f"tz_{city.replace(' ', '_')}"
            sum_id = f"sum_{city.replace(' ', '_')}"
            nodes.append({"id": pop_id, "tool": "get_population",
                          "args": {"city": city}, "deps": []})
            nodes.append({"id": tz_id, "tool": "get_timezone",
                          "args": {"city": city}, "deps": []})
            nodes.append({"id": sum_id, "tool": "summarize_city",
                          "args": {"city": city}, "deps": []})
            fetch_ids_by_city[city] = [pop_id, tz_id, sum_id]
        all_ids = [nid for ids in fetch_ids_by_city.values() for nid in ids]
        nodes.append({
            "id": "aggregate", "tool": "aggregate_report",
            "args": {"cities": cities}, "deps": all_ids,
        })
        return {"nodes": nodes}

    def _compose_report(self, user: str) -> str:
        # Extract the facts the aggregator was handed (stored as JSON inside user prompt)
        lines = ["# City Comparison Report", ""]
        for c in CITY_FACTS:
            if c in user.lower():
                f = CITY_FACTS[c]
                lines.append(
                    f"- **{c.title()}** ({f['country']}): "
                    f"{f['population']:,} people, timezone {f['timezone']}."
                )
        return "\n".join(lines)


# The provider used by LLM-backed tools. Defaults to the offline mock;
# switch with set_provider("anthropic") or set_provider(AnthropicProvider(...)).
llm: LLMProvider = MockProvider()


def set_provider(backend: "str | LLMProvider") -> LLMProvider:
    """Select the LLM backend used by the LLM-backed tools.

    Accepts "mock", "anthropic", or any LLMProvider instance. Returns the
    active provider so callers can keep a reference (e.g. to pass to the
    Orchestrator).
    """
    global llm
    if isinstance(backend, LLMProvider):
        llm = backend
    elif backend == "anthropic":
        llm = AnthropicProvider()
    elif backend == "mock":
        llm = MockProvider()
    else:
        raise ValueError(f"Unknown backend {backend!r} — use 'mock' or 'anthropic'.")
    return llm


# =============================================================================
# 2. Typed tools with Pydantic
# =============================================================================

# TypedTool: adapter between Python callables and LLM-visible JSON Schema tool definitions
@dataclass
class TypedTool:
    name: str
    description: str
    args_model: type[BaseModel]     # Pydantic model defining the arg schema
    fn: Callable[..., Any]
    cost_hint: float = 0.0          # relative cost for budget accounting

    def schema(self) -> dict:
        """Shape expected by Anthropic/OpenAI tool-use APIs."""
        schema = self.args_model.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    def validate_args(self, raw_args: dict) -> tuple[BaseModel | None, str | None]:
        try:
            return self.args_model(**raw_args), None
        except ValidationError as e:
            return None, f"ValidationError: {e.errors()}"

    def run(self, raw_args: dict) -> Any:
        args, err = self.validate_args(raw_args)
        if err is not None:
            raise ValueError(err)
        return self.fn(**args.model_dump())


# Arg models -----------------------------------------------------------------

class CityArgs(BaseModel):
    city: str = Field(..., description="Name of the city, e.g. 'Paris'.")


class AggregateArgs(BaseModel):
    cities: list[str] = Field(..., description="Cities to include in the report.")


# Tool implementations --------------------------------------------------------

def _fact(city: str, key: str) -> Any:
    facts = CITY_FACTS.get(city.lower())
    if facts is None:
        raise KeyError(f"Unknown city: {city!r}")
    return facts[key]


def _safe_json_loads(raw: str) -> dict:
    """Tolerant JSON parse: strips code fences / prose, raises ValueError on failure."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        cleaned = _extract_json_text(raw or "")
        if not cleaned:
            raise ValueError(f"LLM returned no JSON payload (got {raw!r})")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned malformed JSON: {e}. Raw={raw!r}") from e


def get_population(city: str) -> int:
    return _fact(city, "population")


def get_timezone(city: str) -> str:
    return _fact(city, "timezone")


def summarize_city(city: str) -> str:
    # Uses the LLM in "summarizer" role — one LLM call per city.
    raw = llm.complete(
        system="You summarize a city in one sentence. Return JSON: {\"summary\": \"...\"}",
        user=f"City: {city}",
        role="summarizer",
    )
    data = _safe_json_loads(raw)
    return data.get("summary", f"{city.title()} is a notable city.")


def aggregate_report(cities: list[str]) -> str:
    # Also an LLM-backed tool; the aggregator composes the final report.
    payload = {c: CITY_FACTS.get(c.lower(), {}) for c in cities}
    raw = llm.complete(
        system="Compose a markdown report. Return JSON: {\"report\": \"...\"}",
        user=f"Facts: {json.dumps(payload)}",
        role="aggregator",
    )
    data = _safe_json_loads(raw)
    return data.get("report", "")


TOOLS: dict[str, TypedTool] = {
    "get_population": TypedTool(
        name="get_population", description="Look up a city's population.",
        args_model=CityArgs, fn=get_population, cost_hint=0.1,
    ),
    "get_timezone": TypedTool(
        name="get_timezone", description="Look up a city's timezone.",
        args_model=CityArgs, fn=get_timezone, cost_hint=0.1,
    ),
    "summarize_city": TypedTool(
        name="summarize_city",
        description="Produce a one-sentence narrative summary of a city.",
        args_model=CityArgs, fn=summarize_city, cost_hint=1.0,   # LLM call → costlier
    ),
    "aggregate_report": TypedTool(
        name="aggregate_report",
        description="Compose the final markdown comparison report.",
        args_model=AggregateArgs, fn=aggregate_report, cost_hint=2.0,
    ),
}


# =============================================================================
# 3. The plan as a DAG
# =============================================================================

class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PlanNode:
    id: str
    tool: str
    args: dict
    deps: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    attempt: int = 0


@dataclass
class PlanDAG:
    nodes: dict[str, PlanNode] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> "PlanDAG":
        dag = cls()
        for n in data["nodes"]:
            dag.nodes[n["id"]] = PlanNode(
                id=n["id"], tool=n["tool"],
                args=n["args"], deps=n.get("deps", []),
            )
        dag.validate()
        return dag

    def validate(self) -> None:
        # All deps must exist
        for n in self.nodes.values():
            for d in n.deps:
                if d not in self.nodes:
                    raise ValueError(f"Node {n.id!r} depends on missing node {d!r}")
        # No cycles (simple DFS)
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(nid: str) -> None:
            if nid in stack:
                raise ValueError(f"Cycle detected at {nid!r}")
            if nid in visited:
                return
            stack.add(nid)
            for d in self.nodes[nid].deps:
                dfs(d)
            stack.discard(nid)
            visited.add(nid)

        for nid in self.nodes:
            dfs(nid)

    def ready_nodes(self) -> list[PlanNode]:
        """Nodes whose deps are all DONE and are themselves PENDING."""
        out = []
        for n in self.nodes.values():
            if n.status != NodeStatus.PENDING:
                continue
            if all(self.nodes[d].status == NodeStatus.DONE for d in n.deps):
                out.append(n)
        return out

    def is_done(self) -> bool:
        return all(n.status in (NodeStatus.DONE, NodeStatus.FAILED)
                   for n in self.nodes.values())

    def any_failed(self) -> bool:
        return any(n.status == NodeStatus.FAILED for n in self.nodes.values())

    def aggregate_node(self) -> Optional[PlanNode]:
        """Return the capstone aggregate_report node.

        Mock plans use id ``aggregate``; real LLM planners often mirror the tool
        name (e.g. ``aggregate_report``). Resolve by tool, not hard-coded id.
        """
        if "aggregate" in self.nodes and self.nodes["aggregate"].tool == "aggregate_report":
            return self.nodes["aggregate"]
        matches = [n for n in self.nodes.values() if n.tool == "aggregate_report"]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"DAG has multiple aggregate_report nodes: {[n.id for n in matches]}"
            )
        return None

    def plot(self, filename: str = "plan_dag.png", title: str = "Plan DAG"):
        """
        Render the DAG as a left-to-right layered diagram and save it as a PNG.

        Nodes are laid out in vertical columns ("waves"): a node's wave is one
        past the deepest of its dependencies, so each column is exactly the set
        of nodes `execute_dag` can run in parallel once the columns to its left
        are done. Each node is a card tinted by status; a glyph badge repeats
        the status so state is never encoded by color alone.

        Args:
            filename: Output filename for the PNG image.
            title: Figure title.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
        from matplotlib.path import Path

        # Status → (color, glyph, label). Glyph badge keeps state readable
        # without color (colorblind / grayscale print).
        STATUS_STYLE = {
            NodeStatus.PENDING: ("#898781", "○", "pending"),
            NodeStatus.RUNNING: ("#2a78d6", "▶", "running"),
            NodeStatus.DONE:    ("#0ca30c", "✓", "done"),
            NodeStatus.FAILED:  ("#d03b3b", "✕", "failed"),
        }
        INK, INK_SOFT, EDGE_GRAY = "#0b0b0b", "#52514e", "#b3b2ab"

        def tint(color: str, strength: float = 0.10) -> tuple:
            """Light wash of a color over white (cards stay readable)."""
            r, g, b = to_rgb(color)
            return tuple(1 - strength * (1 - c) for c in (r, g, b))

        # --- 1. Longest-path layering: wave(n) = 1 + max(wave of deps) ---
        wave: dict[str, int] = {}

        def _wave(nid: str) -> int:
            if nid not in wave:
                deps = self.nodes[nid].deps
                wave[nid] = 1 + max((_wave(d) for d in deps), default=-1)
            return wave[nid]

        for nid in self.nodes:
            _wave(nid)

        n_waves = max(wave.values()) + 1
        columns: list[list[str]] = [[] for _ in range(n_waves)]
        for nid, w in wave.items():
            columns[w].append(nid)

        # Order wave 0 by tool (groups get_population / get_timezone / ...);
        # later waves follow the average position of their deps (fewer crossings)
        columns[0].sort(key=lambda i: (self.nodes[i].tool, i))
        rank = {nid: k for k, nid in enumerate(columns[0])}
        for col in columns[1:]:
            col.sort(key=lambda i: (
                sum(rank.get(d, 0) for d in self.nodes[i].deps)
                / max(len(self.nodes[i].deps), 1), i))
            rank.update({nid: k for k, nid in enumerate(col)})

        # --- 2. Geometry (data units; aspect locked so cards keep their shape) ---
        CARD_W, CARD_H, DX, DY = 3, 0.78, 4.2, 1.08
        tallest = max(len(col) for col in columns)
        pos: dict[str, tuple[float, float]] = {}
        for w, col in enumerate(columns):
            top = (len(col) - 1) * DY / 2.0          # center each column vertically
            for k, nid in enumerate(col):
                pos[nid] = (w * DX, top - k * DY)

        y_top = (tallest - 1) * DY / 2.0 + CARD_H / 2.0
        fig_w = 1.8 + (n_waves - 1) * DX * 0.72 + CARD_W * 0.72
        fig_h = 2.0 + (tallest - 1) * DY * 0.72 + CARD_H * 0.72
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")

        # --- 3. Edges: smooth S-curves, recessive gray, under the cards ---
        # Arrivals are spread along the target's left edge (sorted by source
        # height) so heavy fan-ins don't collapse into one point.
        for node in self.nodes.values():
            x1, y1 = pos[node.id]
            incoming = sorted(node.deps, key=lambda d: -pos[d][1])
            spread = min(0.14, CARD_H * 0.7 / max(len(incoming), 1))
            for k, dep in enumerate(incoming):
                x0, y0 = pos[dep]
                y_in = y1 + (len(incoming) - 1) * spread / 2 - k * spread
                p0 = (x0 + CARD_W / 2 + 0.05, y0)
                p3 = (x1 - CARD_W / 2 - 0.16, y_in)
                xm = (p0[0] + p3[0]) / 2.0
                path = Path([p0, (xm, y0), (xm, y_in), p3],
                            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                ax.add_patch(FancyArrowPatch(
                    path=path, arrowstyle="-|>", mutation_scale=11,
                    lw=1.1, color=EDGE_GRAY, shrinkA=0, shrinkB=0, zorder=1))

        # --- 4. Node cards: status-tinted box + glyph badge + id / tool text ---
        for nid, (x, y) in pos.items():
            node = self.nodes[nid]
            color, glyph, _ = STATUS_STYLE[node.status]
            ax.add_patch(FancyBboxPatch(
                (x - CARD_W / 2, y - CARD_H / 2), CARD_W, CARD_H,
                boxstyle="round,pad=0.04,rounding_size=0.16",
                facecolor=tint(color), edgecolor=color, lw=1.4, zorder=2))
            bx = x - CARD_W / 2 + 0.34                # glyph badge, left side
            ax.add_patch(Circle((bx, y), 0.17, facecolor=color,
                                edgecolor="none", zorder=3))
            ax.text(bx, y, glyph, ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold", zorder=4)
            tx = x - CARD_W / 2 + 0.62
            ax.text(tx, y + 0.15, node.id, ha="left", va="center",
                    fontsize=10, fontweight="bold", color=INK, zorder=4)
            ax.text(tx, y - 0.18, node.tool, ha="left", va="center",
                    fontsize=8, family="monospace", color=INK_SOFT, zorder=4)

        # --- 5. Wave headers, title, legend ---
        for w in range(n_waves):
            ax.text(w * DX, y_top + 0.42, f"wave {w}", ha="center", va="bottom",
                    fontsize=9, color="#898781")
        done = sum(n.status is NodeStatus.DONE for n in self.nodes.values())
        ax.text(-CARD_W / 2, y_top + 1.15, title, ha="left", va="bottom",
                fontsize=13, fontweight="bold", color=INK)
        ax.text(-CARD_W / 2, y_top + 0.88,
                f"{len(self.nodes)} nodes · {n_waves} waves · {done} done",
                ha="left", va="bottom", fontsize=9, color=INK_SOFT)

        present = [s for s in STATUS_STYLE
                   if any(n.status is s for n in self.nodes.values())]
        handles = [Line2D([], [], marker="o", linestyle="", markersize=8,
                          markerfacecolor=STATUS_STYLE[s][0], markeredgecolor="none",
                          label=f"{STATUS_STYLE[s][1]} {STATUS_STYLE[s][2]}")
                   for s in present]
        fig.legend(handles=handles, loc="lower center", ncol=max(len(handles), 1),
                   frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.6)

        ax.set_xlim(-CARD_W / 2 - 0.4, (n_waves - 1) * DX + CARD_W / 2 + 0.4)
        ax.set_ylim(-y_top - 0.7, y_top + 1.7)
        plt.savefig(filename, bbox_inches="tight", dpi=200,
                    facecolor=fig.get_facecolor())
        plt.close(fig)


# =============================================================================
# 4. Parallel execution with asyncio
# =============================================================================

# Level-synchronous DAG executor: run all ready nodes each wave until done or blocked
MAX_CONCURRENT = 5  # cap concurrent tool/LLM calls — tune per environment and rate limits
MAX_NODE_RETRIES = 2  # retries per node, TRANSIENT errors only (see classify_error)


async def execute_dag(dag: PlanDAG, tools: dict[str, TypedTool],
                       on_step: Callable[[PlanNode, float], None] | None = None,
                       max_retries: int = MAX_NODE_RETRIES):
    """Walk the DAG in waves; each wave runs every currently-ready node in parallel.

    Node failures are captured on the node rather than raised, so one bad node
    never aborts its wave. Failures classified TRANSIENT (rate limits, timeouts)
    are retried with backoff before being recorded; every other error class is
    recorded on the first attempt and left for the orchestrator to classify —
    re-planning, not retrying, is the answer to a missing entity.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def run_node(node: PlanNode) -> None:
        node.status = NodeStatus.RUNNING
        async with semaphore:
            t0 = time.time()
            try:
                tool = tools[node.tool]

                async def attempt() -> Any:
                    # node.attempt counts real invocations, so a node that
                    # succeeded on its third try says so in the trace.
                    node.attempt += 1
                    # Sync tools run in a thread pool so other nodes can proceed
                    return await asyncio.to_thread(tool.run, node.args)

                node.result = await retry_with_backoff(attempt, max_retries=max_retries)
                node.status = NodeStatus.DONE
            except Exception as exc:
                node.error = f"{type(exc).__name__}: {exc}"
                node.status = NodeStatus.FAILED
            finally:
                if on_step:
                    on_step(node, time.time() - t0)

    while not dag.is_done():
        ready = dag.ready_nodes()
        if not ready:
            # Remaining PENDING nodes depend on FAILED ancestors — no forward progress
            break
        await asyncio.gather(*(run_node(n) for n in ready))


# =============================================================================
# 5. Multi-tier memory
# =============================================================================

def _bag(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _sim(a: str, b: str) -> float:
    ba, bb = _bag(a), _bag(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)   # Jaccard — fine for teaching


@dataclass
class Memory:
    content: str
    kind: str                           # "episodic" | "semantic"
    metadata: dict = field(default_factory=dict)
    embedding: Optional[Any] = None     # Store embedding if using real embeddings


class MemoryStore:
    """Similarity store with optional real embeddings support."""

    def __init__(self, use_embeddings: bool = USE_REAL_EMBEDDINGS) -> None:
        self._items: list[Memory] = []
        self._use_embeddings = use_embeddings
        self._model = None

        if self._use_embeddings:
            try:
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers is not installed")
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                print("MemoryStore: Using sentence-transformers embeddings (all-MiniLM-L6-v2)")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                print("Falling back to Jaccard similarity")
                self._use_embeddings = False
        else:
            print("MemoryStore: Using Jaccard similarity")

    def add(self, content: str, kind: str, **meta) -> None:
        embedding = None
        if self._use_embeddings and self._model is not None:
            embedding = self._model.encode(content, convert_to_tensor=False)
        self._items.append(Memory(content=content, kind=kind, metadata=meta, embedding=embedding))

    def retrieve(self, query: str, k: int = 3, kind: str | None = None) -> list[Memory]:
        pool = self._items if kind is None else [m for m in self._items if m.kind == kind]

        if self._use_embeddings and self._model is not None:
            # Use cosine similarity with embeddings
            query_emb = self._model.encode(query, convert_to_tensor=False)
            scored = []
            for m in pool:
                if m.embedding is not None:
                    # Cosine similarity
                    sim = np.dot(query_emb, m.embedding) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(m.embedding)
                    )
                    scored.append((m, float(sim)))
                else:
                    # Fallback to Jaccard if embedding missing
                    scored.append((m, _sim(query, m.content)))
        else:
            # Use Jaccard similarity
            scored = [(m, _sim(query, m.content)) for m in pool]

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, score in scored[:k] if score > 0]


@dataclass
class WorkingMemory:
    """Stays in context for the current task."""
    goal: str
    plan_summary: str = ""
    recent_results: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        parts = [f"Goal: {self.goal}"]
        if self.plan_summary:
            parts.append(f"Plan: {self.plan_summary}")
        if self.recent_results:
            parts.append("Recent results:")
            for r in self.recent_results[-5:]:
                parts.append(f"  - {r}")
        return "\n".join(parts)


def build_context(working: WorkingMemory, store: MemoryStore,
                  budget_chars: int = 4000) -> str:
    """Assemble working memory + retrieved memories, respecting a char budget."""
    pieces: list[str] = [working.to_prompt()]
    used = len(pieces[0])

    # Episodic first (past similar tasks), then semantic (facts)
    for kind in ("episodic", "semantic"):
        for m in store.retrieve(working.goal, k=3, kind=kind):
            snippet = f"[{kind}] {m.content}"
            if used + len(snippet) + 1 > budget_chars:
                return "\n".join(pieces) + "\n(...truncated at budget...)"
            pieces.append(snippet)
            used += len(snippet) + 1

    return "\n".join(pieces)


# =============================================================================
# 6. Verification hierarchy
# =============================================================================

# --- Verification: cheap deterministic checks before expensive LLM judge ---
@dataclass
class Verdict:
    passed: bool
    tier: str                     # "deterministic" | "llm_judge"
    reason: str = ""


# A deterministic check maps a report to a Verdict. Swap in your own to verify
# something other than "does the text mention these terms" — golden-file diffs
# and human review are additional tiers, not implemented here.
ReportCheck = Callable[[str], Verdict]


def deterministic_check_report(report: str, required_terms: list[str]) -> Verdict:
    """The report must be non-empty and mention every required term."""
    if not report or not report.strip():
        return Verdict(False, "deterministic", "Empty report.")
    missing = [c for c in required_terms if c.lower() not in report.lower()]
    if missing:
        return Verdict(False, "deterministic", f"Missing required terms: {missing}")
    return Verdict(True, "deterministic", "All required terms present.")


def llm_judge_report(report: str, goal: str, provider: LLMProvider) -> Verdict:
    """Ask the LLM to grade the report against the goal. Returns JSON {pass, reason}."""
    raw = provider.complete(
        system=(
            "You are a strict grader. Given a GOAL and a REPORT, decide whether "
            "the report satisfies the goal. Respond ONLY with JSON: "
            '{"pass": true|false, "reason": "..."}.'
        ),
        user=f"GOAL:\n{goal}\n\nREPORT:\n{report}",
        role="critic",
    )
    try:
        data = _safe_json_loads(raw)
        return Verdict(bool(data["pass"]), "llm_judge", data.get("reason", ""))
    except Exception as e:
        return Verdict(False, "llm_judge", f"Judge output malformed: {e}")


def verify_report(report: str, goal: str, required_terms: list[str],
                  provider: LLMProvider,
                  check: Optional[ReportCheck] = None) -> Verdict:
    det = check(report) if check else deterministic_check_report(report, required_terms)
    if not det.passed:
        return det                # Cheap tier caught it — don't bother the LLM
    # Deterministic passed → escalate to LLM judge for subjective quality
    return llm_judge_report(report, goal, provider)


# =============================================================================
# 7. Multi-agent roles (Planner / Worker / Critic)
# =============================================================================

PLANNER_SYSTEM = """You are a Planner agent. Given a GOAL, produce a dependency graph
of tool calls that will satisfy it.

Output ONLY JSON in this shape (no prose, no markdown):
{"nodes": [{"id": "...", "tool": "...", "args": {...}, "deps": [...]}, ...]}

Nodes may run in parallel if their `deps` are empty or already satisfied.
Available tools (name and schema):
<<TOOLS_JSON>>
"""


class PlannerAgent:
    def __init__(self, provider: LLMProvider, tools: dict[str, TypedTool]):
        self._provider = provider
        self._tools = tools

    def plan(self, goal: str) -> PlanDAG:
        tools_json = json.dumps([t.schema() for t in self._tools.values()], indent=2)
        system = PLANNER_SYSTEM.replace("<<TOOLS_JSON>>", tools_json)
        raw = self._provider.complete(system=system, user=f"GOAL: {goal}",
                                       role="planner")
        return PlanDAG.from_json(_safe_json_loads(raw))

    def replan(self, goal: str, failed_dag: PlanDAG) -> PlanDAG:
        """Re-plan after failures, providing context about what failed."""
        tools_json = json.dumps([t.schema() for t in self._tools.values()], indent=2)
        system = PLANNER_SYSTEM.replace("<<TOOLS_JSON>>", tools_json)

        # Collect failure information
        failures = []
        for node in failed_dag.nodes.values():
            if node.status == NodeStatus.FAILED:
                failures.append(f"Node {node.id} (tool={node.tool}, args={node.args}) failed: {node.error}")

        failure_context = "\n".join(failures)
        user_prompt = (
            f"GOAL: {goal}\n\n"
            f"Previous plan failed with these errors:\n{failure_context}\n\n"
            f"Please provide an amended plan that avoids these failures."
        )

        raw = self._provider.complete(system=system, user=user_prompt, role="replanner")
        return PlanDAG.from_json(_safe_json_loads(raw))


class CriticAgent:
    def __init__(self, provider: LLMProvider):
        self._provider = provider

    def judge(self, report: str, goal: str, required_terms: list[str],
              check: Optional[ReportCheck] = None) -> Verdict:
        return verify_report(report, goal, required_terms, self._provider, check=check)


class WorkerAgent:
    """Runs the DAG produced by the Planner."""

    def __init__(self, tools: dict[str, TypedTool]):
        self._tools = tools

    async def execute(self, dag: PlanDAG,
                      on_step: Callable[[PlanNode, float], None] | None = None) -> None:
        await execute_dag(dag, self._tools, on_step=on_step)


# =============================================================================
# 8. Budgeting and recovery strategies
# =============================================================================

@dataclass
class BudgetMulti:
    max_tokens: int = 50_000
    max_tool_calls: int = 30
    max_wall_seconds: float = 60.0
    max_cost_usd: float = 0.50

    tokens_used: int = 0
    tool_calls_used: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def has_room(self) -> bool:
        return (
            self.tokens_used < self.max_tokens
            and self.tool_calls_used < self.max_tool_calls
            and self.elapsed() < self.max_wall_seconds
            and self.cost_usd < self.max_cost_usd
        )

    def pressure(self) -> float:
        return max(
            self.tokens_used / self.max_tokens,
            self.tool_calls_used / self.max_tool_calls,
            self.elapsed() / self.max_wall_seconds,
            self.cost_usd / self.max_cost_usd,
        )

    def charge(self, *, tokens: int = 0, cost: float = 0.0,
               tool_calls: int = 0) -> None:
        self.tokens_used += tokens
        self.cost_usd += cost
        self.tool_calls_used += tool_calls


class ErrorClass(str, Enum):
    TRANSIENT = "transient"       # rate limit, timeout → backoff + retry
    TOOL_MISUSE = "tool_misuse"   # bad args → let LLM self-correct
    MISSING_INFO = "missing_info" # 404, not found → re-plan
    FATAL = "fatal"               # security violation, policy → halt


def classify_error(err: str) -> ErrorClass:
    e = err.lower()
    if any(s in e for s in ("timeout", "rate limit", "503", "502", "connection")):
        return ErrorClass.TRANSIENT
    if "validationerror" in e or ("missing" in e and "arg" in e):
        return ErrorClass.TOOL_MISUSE
    if "keyerror" in e or "not found" in e or "unknown" in e:
        return ErrorClass.MISSING_INFO
    return ErrorClass.FATAL


async def retry_with_backoff(fn: Callable[[], Any], *,
                              max_retries: int = 3) -> Any:
    """Retry a callable on TRANSIENT errors with exponential backoff + jitter."""
    for attempt in range(max_retries + 1):
        try:
            return await fn() if asyncio.iscoroutinefunction(fn) else fn()
        except Exception as exc:
            err_cls = classify_error(str(exc))
            if err_cls != ErrorClass.TRANSIENT or attempt >= max_retries:
                raise
            wait = min(2 ** attempt + random.uniform(0, 1), 10)
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


# =============================================================================
# 9. Structured observability and tracing
# =============================================================================

@dataclass
class TraceEvent:
    step_id: str
    parent_id: Optional[str]
    timestamp: float
    role: str                         # "planner" | "worker" | "critic"
    action: dict
    result: Any
    latency_ms: int
    tokens: int
    cost_usd: float
    budget_pressure: float
    verdict: Optional[dict] = None    # {passed, tier, reason}

    def to_json(self) -> dict:
        return {
            "step_id": self.step_id,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "role": self.role,
            "action": self.action,
            "result": _truncate(self.result, 200),
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 6),
            "budget_pressure": round(self.budget_pressure, 3),
            "verdict": self.verdict,
        }


def _truncate(x: Any, n: int) -> Any:
    s = str(x)
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n} chars)"


class Tracer:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, **kwargs) -> TraceEvent:
        ev = TraceEvent(**kwargs)
        self.events.append(ev)
        return ev

    def dump(self) -> list[dict]:
        return [e.to_json() for e in self.events]

    def new_step_id(self, prefix: str = "s") -> str:
        return f"{prefix}-{len(self.events):04d}"


# =============================================================================
# 10. The Orchestrator: composition root
# =============================================================================

# Illustrative blended rate, NOT a measured price. Every token count in this
# module is estimated (characters // 4 for prompts, cost_hint for tools), so
# every cost derived from it is an estimate of an estimate — useful as a
# relative signal between runs, not as an invoice. Production wires up the
# provider's usage metadata and a per-model rate instead.
COST_PER_1K_TOKENS = 0.003


@dataclass
class RunResult:
    goal: str
    report: Optional[str]
    verdict: Optional[Verdict]
    dag: PlanDAG
    tracer: Tracer
    budget: BudgetMulti
    status: str                      # "ok" | "failed_verify" | "failed_execute" | "budget" | "replanned"
    replan_count: int = 0            # Track how many times we replanned


class Orchestrator:
    def __init__(self, provider: LLMProvider, tools: dict[str, TypedTool],
                 memory: MemoryStore, max_replans: int = 2):
        self._provider = provider
        self._tools = tools
        self._memory = memory
        self._max_replans = max_replans
        self.planner = PlannerAgent(provider, tools)
        self.worker = WorkerAgent(tools)
        self.critic = CriticAgent(provider)

    async def run(self, goal: str, required_terms: list[str],
                   budget: Optional[BudgetMulti] = None,
                   check: Optional[ReportCheck] = None) -> RunResult:
        """Plan, execute, verify. ``check`` overrides the deterministic tier so
        the orchestrator can verify goals that aren't keyword-shaped."""
        budget = budget or BudgetMulti()
        tracer = Tracer()
        replan_count = 0
        plan_step_id = None

        # --- 1. Plan (with re-planning loop) ---
        dag = None
        for attempt in range(self._max_replans + 1):
            t0 = time.time()

            if attempt == 0:
                # Initial plan
                dag = self.planner.plan(goal)
                plan_type = "initial_plan"
            else:
                # Re-plan after failure
                dag = self.planner.replan(goal, dag)
                plan_type = "replan"
                replan_count += 1

            plan_latency = int((time.time() - t0) * 1000)
            est_tokens = (len(goal) + sum(len(n.tool) + len(json.dumps(n.args)) for n in dag.nodes.values())) // 4
            budget.charge(tokens=est_tokens,
                          cost=est_tokens * COST_PER_1K_TOKENS / 1000)
            # Hold the id so worker/critic events can point at the plan that
            # actually produced them. Deriving it from a count of planner events
            # breaks after a replan, because step ids number by total events.
            plan_step_id = tracer.new_step_id("plan")
            tracer.record(
                step_id=plan_step_id, parent_id=None,
                timestamp=time.time(), role="planner",
                action={"type": plan_type, "goal": goal, "attempt": attempt},
                result={"n_nodes": len(dag.nodes)},
                latency_ms=plan_latency, tokens=est_tokens,
                cost_usd=est_tokens * COST_PER_1K_TOKENS / 1000,
                budget_pressure=budget.pressure(),
            )

            # --- 2. Execute with per-step tracing ---
            def on_step(node: PlanNode, wall_s: float, _parent=plan_step_id) -> None:
                tool = self._tools[node.tool]
                # A node that was retried really did call the tool more than
                # once; bill every attempt or the budget under-reports.
                attempts = max(1, node.attempt)
                toks = int(max(20, tool.cost_hint * 200)) * attempts
                budget.charge(tokens=toks,
                              cost=toks * COST_PER_1K_TOKENS / 1000,
                              tool_calls=attempts)
                tracer.record(
                    step_id=tracer.new_step_id("w"), parent_id=_parent,
                    timestamp=time.time(), role="worker",
                    action={"type": "tool_call", "tool": node.tool, "args": node.args},
                    result=node.result if node.status == NodeStatus.DONE else node.error,
                    latency_ms=int(wall_s * 1000),
                    tokens=toks, cost_usd=toks * COST_PER_1K_TOKENS / 1000,
                    budget_pressure=budget.pressure(),
                )

            try:
                await self.worker.execute(dag, on_step=on_step)
            except Exception as exc:
                if attempt < self._max_replans and budget.has_room():
                    continue  # Try re-planning
                return RunResult(goal=goal, report=None, verdict=None, dag=dag,
                                 tracer=tracer, budget=budget,
                                 status=f"failed_execute:{exc}", replan_count=replan_count)

            # Check if we need to re-plan
            if dag.any_failed():
                if attempt < self._max_replans and budget.has_room():
                    # Classify errors to decide if re-planning makes sense
                    should_replan = False
                    for node in dag.nodes.values():
                        if node.status == NodeStatus.FAILED:
                            err_class = classify_error(node.error or "")
                            if err_class == ErrorClass.MISSING_INFO:
                                should_replan = True
                                break

                    if should_replan:
                        continue  # Re-plan

                # Can't or shouldn't re-plan
                return RunResult(goal=goal, report=None, verdict=None, dag=dag,
                                 tracer=tracer, budget=budget, status="failed_execute",
                                 replan_count=replan_count)

            # Success - break out of re-planning loop
            break

        agg_node = dag.aggregate_node()
        # A present-but-failed aggregate node has no report to verify; treat it
        # as an execution failure rather than letting an empty report fall
        # through to the verifier and get misreported as failed_verify.
        if agg_node is None or agg_node.status != NodeStatus.DONE:
            return RunResult(goal=goal, report=None, verdict=None, dag=dag,
                             tracer=tracer, budget=budget, status="failed_execute",
                             replan_count=replan_count)

        report = agg_node.result

        # Pressure-aware degradation: skip LLM judge if budget is tight
        t0 = time.time()
        if budget.pressure() > 0.9:
            verdict = (check(report) if check
                       else deterministic_check_report(report, required_terms))
        else:
            verdict = self.critic.judge(report, goal, required_terms, check=check)
        judge_latency = int((time.time() - t0) * 1000)

        tracer.record(
            step_id=tracer.new_step_id("c"), parent_id=plan_step_id,
            timestamp=time.time(), role="critic",
            action={"type": "verify"},
            result={"passed": verdict.passed, "tier": verdict.tier},
            latency_ms=judge_latency, tokens=100,
            cost_usd=100 * COST_PER_1K_TOKENS / 1000,
            budget_pressure=budget.pressure(),
            verdict={"passed": verdict.passed, "tier": verdict.tier,
                     "reason": verdict.reason},
        )

        # Stash the outcome in episodic memory for future runs
        self._memory.add(
            f"Task: {goal[:120]} → outcome: "
            f"{'pass' if verdict.passed else 'fail'} ({verdict.reason})",
            kind="episodic",
        )

        status = "ok" if verdict.passed else "failed_verify"
        if replan_count > 0:
            status = f"{status}_replanned_{replan_count}x"
        if not budget.has_room():
            status = "budget"
        return RunResult(goal=goal, report=report, verdict=verdict, dag=dag,
                         tracer=tracer, budget=budget, status=status,
                         replan_count=replan_count)


__all__ = [
    # Providers
    "LLMProvider", "AnthropicProvider", "MockProvider", "llm", "set_provider",
    "CITY_FACTS",
    # Tools
    "TypedTool", "CityArgs", "AggregateArgs", "TOOLS",
    "get_population", "get_timezone", "summarize_city", "aggregate_report",
    # DAG + execution
    "NodeStatus", "PlanNode", "PlanDAG", "execute_dag", "MAX_CONCURRENT",
    "MAX_NODE_RETRIES",
    # Memory
    "Memory", "MemoryStore", "WorkingMemory", "build_context", "USE_REAL_EMBEDDINGS",
    # Verification
    "Verdict", "ReportCheck", "deterministic_check_report", "llm_judge_report",
    "verify_report",
    # Agents
    "PLANNER_SYSTEM", "PlannerAgent", "CriticAgent", "WorkerAgent",
    # Budget + recovery
    "BudgetMulti", "ErrorClass", "classify_error", "retry_with_backoff",
    # Tracing
    "TraceEvent", "Tracer",
    # Orchestration
    "COST_PER_1K_TOKENS", "RunResult", "Orchestrator",
]
