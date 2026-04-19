![GitHub](https://img.shields.io/github/license/DataForScience/LLMs)
[![Twitter @data4sci](https://img.shields.io/twitter/follow/data4sci)](https://twitter.com/intent/follow?screen_name=data4sci)
![GitHub top language](https://img.shields.io/github/languages/top/DataForScience/LLMs)
![GitHub repo size](https://img.shields.io/github/repo-size/DataForScience/LLMs)
![GitHub last commit](https://img.shields.io/github/last-commit/DataForScience/LLMs)

[![Graphs For Science](https://img.shields.io/badge/Graphs_For_Science-Subscribe-blue)](https://graphs4sci.substack.com/)
[![Data Science Briefing](https://img.shields.io/badge/Data_Science_Briefing-Subscribe-blue)](https://data4science.ck.page/a63d4cc8d9)

# Building Agentic Harnesses from Scratch

Educational notebooks demonstrating how to build production-grade agentic systems from first principles using LLMs. Learn the core patterns behind tools like Claude Code, Cursor's agent mode, and autonomous research assistants by implementing them yourself.

These notebooks teach you to build the scaffolding that transforms an LLM from a text generator into an agent that can call tools, maintain state, recover from errors, and pursue goals across multiple steps.

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

### Notebook 2: State-of-the-Art Agentic Harness
Upgrade every component toward production-grade systems like Claude Code, Devin, or modern research agents.

**Advanced topics:**
- **Typed tools with Pydantic** — Auto-generated JSON schemas and robust validation
- **DAG orchestration** — Parallel execution of independent tasks instead of sequential processing
- **Multi-tier memory** — Working, episodic, and semantic memory with retrieval
- **Verification hierarchy** — Deterministic tests → output diffing → LLM-as-judge → human review
- **Multi-agent roles** — Planner / Worker / Critic specialization for robustness
- **Multi-dimensional budgeting** — Graceful degradation under token, time, and cost constraints
- **Structured tracing** — Full observability and replay capability

**What you'll build:** A city comparison agent that demonstrates natural parallelism, final aggregation, and verifiable outputs.

## Notebooks

- **[01 - Basic Agentic Harness.ipynb](./01%20-%20Basic%20Agentic%20Harness.ipynb)** — Start here to understand the fundamentals
- **[02 - Advanced Agentic Harness.ipynb](./02%20-%20Advanced%20Agentic%20Harness.ipynb)** — Production patterns and advanced techniques

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

To use real LLM APIs, set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then change `BACKEND = "mock"` to `BACKEND = "anthropic"` in the notebook.

### 3) Launch notebooks

```bash
jupyter notebook
```

## Repository Structure

```
LLMs/
├── 01 - Basic Agentic Harness.ipynb     # Notebook 1: Fundamentals
├── 02 - Advanced Agentic Harness.ipynb  # Notebook 2: Production patterns
├── Template.ipynb                        # Blank template for experiments
├── data/                                 # Logos and assets
│   ├── D4Sci_logo_ball.png
│   ├── D4Sci_logo_full.png
│   └── bgoncalves.png
├── old/                                  # Archive of previous experiments
├── d4sci.mplstyle                        # Custom matplotlib style
├── pyproject.toml                        # Dependency manifest (for `uv sync`)
├── uv.lock                               # Lock file for reproducible builds
└── LICENSE                               # MIT License
```

## Learning Path

1. **Start with Notebook 1** (`01 - Basic Agentic Harness.ipynb`) to understand the core concepts
2. **Progress to Notebook 2** (`02 - Advanced Agentic Harness.ipynb`) for production-grade patterns
3. **Experiment** with the `Template.ipynb` to build your own agents

## Key Features

- **No API keys required** — All notebooks run in mock mode with rule-based LLM stand-ins
- **Fully reproducible** — Deterministic outputs for teaching and debugging
- **Production-ready patterns** — Learn the same techniques used in Claude Code, Cursor, and Devin
- **Hands-on implementation** — Build everything from scratch to understand every design decision

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