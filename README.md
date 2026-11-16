# StaffPilot Crew

A multi agent copilot for support staffing decisions. IE5374 Applied Generative AI, final project.

**Status:** proposal stage. The proposal is in [docs/PROPOSAL.md](docs/PROPOSAL.md).

## What it will do

A staffing manager types an open ended request, such as "Sign ups in the USA came in 20% above forecast in March. What should we do, and what does it cost?". Five agents work on it together:

| Agent | Role | What it does |
|---|---|---|
| Planner (supervisor) | reasoning | splits the request into steps and asks back when it is unclear |
| Analyst | reasoning | explains the data: seasonality, forecast, who is short or over staffed |
| Optimizer | optimization | runs the integer program and the day by day simulator on alternatives |
| Decision critic | decision making | compares the options, checks every rule and number, sends bad options back |
| Writer | generation | writes the answer for a manager, finance or a country manager |

A router picks a model for each step: Claude Sonnet 5.5, GPT 5 nano, Gemini 3 Flash, or Gemma 4 E4B locally. The graph runs countries in parallel, caps both critique loops (back to the Optimizer, back to the Writer) at 2 retries each, explains scenarios that have no solution, and pauses for the manager's approval before anything is applied. It also plans human and AI capacity together (a share of advertisers handled by an AI assistant) and reviews each monthly update without being asked.

We compared the idea with two commercial tools, NiCE Workforce Management and Anaplan's Workforce Analyst, and chose what to take from them in [docs/RELATED_PRODUCTS.md](docs/RELATED_PRODUCTS.md).

## How this repository relates to the hackathon

The optimizer, the simulator, the data loader and the model backends come from our Module 7 hackathon, [StaffPilot](https://github.com/lequo-neu/staffpilot). That repository is kept exactly as it was submitted; this project installs it as a library, pinned to the submitted commit (see `pyproject.toml`). Both repositories are public, so the install needs no GitHub sign in.

Everything in this repository is new work for the final project: the agents, the LangGraph graph, the router, the interface and the multi step evaluation.

## Planned layout

| Path | Contents |
|---|---|
| `src/staffpilot_crew/` | agents, graph, router and app |
| `tests/` | unit tests |
| `eval/` | multi step questions, labels and results |
| `docs/` | proposal, report, slides |
| `data/raw/` | the course dataset (not committed) |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

Copy `.env.example` to `.env` and add the keys of the providers you want to use. With no key, the app will use Gemma 4 E4B through LM Studio if it is running, and otherwise a deterministic offline backend. The tests and CI always use the deterministic backend, so they need no key and no model download.

## Team

Seven Team-5374, the same team as our Module 7 hackathon.

| Member | Lead on |
|---|---|
| Quoc Hung Le (Kevin) | graph, router, evaluation |
| Hamad Al Sulaimi | Optimizer agent, interface |
| Khoa Tran | Writer agent, role views, evaluation questions |

## AI use

How AI tools were used will be recorded in `docs/AI_USAGE.md`, as in the hackathon.
