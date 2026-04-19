![GitHub](https://img.shields.io/github/license/DataForScience/CrewAI)
[![Twitter @data4sci](https://img.shields.io/twitter/follow/data4sci)](https://twitter.com/intent/follow?screen_name=data4sci)
![GitHub top language](https://img.shields.io/github/languages/top/DataForScience/CrewAI)
![GitHub repo size](https://img.shields.io/github/repo-size/DataForScience/CrewAI)
![GitHub last commit](https://img.shields.io/github/last-commit/DataForScience/CrewAI)

[![Graphs For Science](https://img.shields.io/badge/Graphs_For_Science-Subscribe-blue)](https://graphs4sci.substack.com/)
	[![Data Science Briefing](https://img.shields.io/badge/Data_Science_Briefing-Subscribe-blue)](https://data4science.ck.page/a63d4cc8d9)

# CrewAI for Production-Ready Multi‑Agent Systems

**Code + slides** to accompany the O’Reilly Live Training, [CrewAI for Production‑Ready Multi‑Agent Systems](https://www.oreilly.com/live-events/crewai-for-production-ready-multiagent-systems/0642572289041/) by Data For Science, Inc.

**Want the complete Gumroad package?** Get the full course materials here: **[Purchase on Gumroad](https://gum.new/gum/cmlubht78001m04jx0ttyh1qk)**.


This live training will take developers beyond the basics of LLM interaction into the realm of building sophisticated, autonomous multi-agent systems. While Large Language Models are powerful on their own, their true potential is unlocked when they are orchestrated into teams that can plan, execute, and review complex tasks. CrewAI is quickly becoming state of the art in AI agent development and orchestration.

This live training will help master the CrewAI framework through hand-on examples. We start by building a simple "Hello World" research agent, then quickly advance to constructing a team of financial analysts equipped with custom tools. We then explore advanced concepts like hierarchical delegation—where a manager agent supervises a team—and implement long-term memory so agents "learn" from past executions. Finally, we cover essential production patterns, including human-in-the-loop approval flows for sensitive tasks (like writing code) and separating agent logic from configuration for maintainable software.

## Course Structure
### 1. Foundations of Autonomous Agents
- Core Concepts
- Roles
- Goals
- Backstories

### 2. Tools and Sequential Crews
- Use Search and APIs
- Develop Custom Tools
- Understand Context

### 3. Orchestration & Memory
- Hierarchical Process
- Managers and Task delegation
- RAG
- Vector Databases

### 4. Human-in-the-Loop (HITL)
- Files
- Code development
- HITL Workflows

### 5. Production Patterns
- Extensible configurations
- Supporting multiple LLMs
- Error Handling
- Best Practices

## What you’ll get (and how to use it)

- **5 hands-on notebooks**: start simple, then level up into tools, orchestration, memory/RAG, HITL, and production patterns.
- **Slides**: Detailed PDF slide deck

## Start here

- **Module 1**: [`1. Foundations.ipynb`](./1.%20Foundations.ipynb)
- **Module 2**: [`2. Tools, Sequential, Crews.ipynb`](./2.%20Tools,%20Sequential,%20Crews.ipynb)
- **Module 3**: [`3. Orchestration and Memory.ipynb`](./3.%20Orchestration%20and%20Memory.ipynb)
- **Module 4**: [`4. Human in the Loop.ipynb`](./4.%20Human%20in%20the%20Loop.ipynb)
- **Module 5**: [`5. Production Patterns.ipynb`](./5.%20Production%20Patterns.ipynb)
- **Slides (PDF)**: [`slides/CrewAI.pdf`](./slides/CrewAI.pdf)

## Setup

### 1) Install dependencies (recommended: `uv`)

1. Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create an environment and install dependencies:

```bash
git clone https://github.com/DataForScience/CrewAI.git
cd CrewAI
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### 2) Add API keys

You'll need accounts with the following services. All offer free tiers or pay-as-you-go pricing:

| Service    | Used in         | Where to get it                          |
|------------|-----------------|------------------------------------------|
| Anthropic  | All notebooks   | console.anthropic.com                    |
| Serper     | Modules 2–5     | serper.dev                               |
| VoyageAI   | Module 3 (RAG)  | dash.voyageai.com *(optional)*           |

Once you have your keys, create a `.env` file in the root of the project:

```
ANTHROPIC_API_KEY=sk-ant-...
SERPER_API_KEY=...
VOYAGE_API_KEY=...  
```

### 3) Launch notebooks

```bash
jupyter notebook
```

## Repository Structure

```
CrewAI/
├── 1. Foundations.ipynb                 # Module 1: Agent basics
├── 2. Tools, Sequential, Crews.ipynb    # Module 2: Tools + workflows
├── 3. Orchestration and Memory.ipynb    # Module 3: Orchestration + memory + RAG
├── 4. Human in the Loop.ipynb           # Module 4: HITL patterns
├── 5. Production Patterns.ipynb         # Module 5: Production best practices
├── slides/                               # Slides
│   └── CrewAI.pdf
├── data/                                 # Logos + author image assets
├── config/                               # YAML agent/task config (Module 5)
├── d4sci.mplstyle                        # Custom matplotlib style
├── pyproject.toml                        # Dependency manifest (for `uv sync`)
├── requirements.txt                      # Alternative install path (pip/uv pip)
└── .env                                  # API keys (create this file; do not commit)
```

## Suggested learning path

- Start with **Module 1** and run the notebooks in order (1 → 5).
- If you only want the deck, open `slides/CrewAI.pdf`.

| File                          | Topic                          |
|-------------------------------|--------------------------------|
| `1. Foundations.ipynb`           | Agents, roles, goals           |
| `2. Tools, Sequential, Crews.ipynb` | Tools, multi-agent workflows |
| `3. Orchestration and Memory.ipynb` | Hierarchical crews, memory, RAG |
| `4. Human in the Loop.ipynb`     | Approval flows, safety patterns |
| `5. Production Patterns.ipynb`   | Config, retries, monitoring    |

---

## Estimated API Costs

Running through all notebooks end-to-end costs roughly **$1–3** in API calls, depending on how much you re-run cells. The biggest spend is Module 4 (Human in the Loop), which generates longer outputs during the multi-stage workflow examples.

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