# Final Project Proposal

**Team:** Seven Team-5374, the same team as our Module 7 hackathon
**Members:** Quoc Hung Le (Kevin), Hamad Al Sulaimi, Khoa Tran

## Project title

**StaffPilot Crew: a multi agent copilot for support staffing decisions**

## The problem

A company that supports advertisers in 22 countries must decide every month how many support agents to hire or release in each country. Sign ups change with the season and the market. Hiring takes a month to act, and too few agents means advertisers wait and leave. Today a staffing manager answers questions like these with spreadsheets and guesswork:

* "Sign ups in the USA came in 20% above forecast in March. What should we do, and what does it cost?"
* "We are thinking of launching a campaign in Brazil in Q3. Can our team handle it?"
* "Compare a Q3 growth push in the USA, Brazil and the UAE. Where does it pay?"
* "Why does support in the UAE lose money, and what are our options?"
* "Write a short note to the Vietnam manager that explains why they are hiring."

In our Module 7 hackathon we built StaffPilot: an optimizer that plans the team for each country and month ($479.7M net value, $109.3M more than keeping the January team) and a single LangChain assistant that answers questions about that plan. The final project turns that single assistant into **a team of agents** that can take an open ended request, work out what needs to be done, run the analysis, weigh the options, and write the answer for the right person, while choosing **the best language model for each step**.

## Users

* **Staffing manager:** plans the year and reacts every month.
* **Country managers:** need their own numbers explained.
* **Finance:** needs proof that a change pays.
* **People team:** needs notices that are fair and planned early.

## Related products and what we take from them

Two commercial tools already put AI into staffing decisions. Our full comparison is in the repository, at https://github.com/lequo-neu/staffpilot-crew/blob/main/docs/RELATED_PRODUCTS.md.

* **NiCE Workforce Management and its copilots** plan capacity across human and AI agents for contact centers, from intraday to weeks (source: https://www.nice.com/products/copilots).
* **Anaplan's role based agents** include a Workforce Analyst that finds workforce risks, runs what if scenarios on headcount and carries out changes only after approval (source: https://www.anaplan.com/news/anaplan-introduces-new-suite-role-based-ai-agents/, 9 December 2025).

From them we add two features that fit our data: an **automation share lever** (part of the advertisers handled by an AI assistant, so the plan covers human and AI capacity) and **monitoring of each monthly update** (the agents review new sign ups without being asked and propose next month's changes for approval). Both tools also confirm two choices we had already made: approval before any change, and a record of every step. We leave out what our monthly data cannot support, such as intraday shifts. Neither vendor states which language models it uses or how a number in an answer is traced; measured model routing and traced numbers are what our project adds.

## Approach

### Agents and their roles

| Agent | Course role | What it does | Main tools |
|---|---|---|---|
| **Planner (supervisor)** | reasoning | reads the open ended request, splits it into steps (one per country when several are named), decides which agents to call and in what order, asks back when the request is unclear | task plan, conversation memory |
| **Analyst** | reasoning | explains the data: seasonality, forecast, which countries are short or over staffed, and why; reviews each monthly update against the forecast without being asked | forecast, data summary, RAG over the brief's rules and our assumptions |
| **Optimizer** | optimization | turns a request into a scenario and runs it: integer program per country plus the day by day simulator; tries alternatives (more hires, a service goal, a different policy, a share of advertisers handled by an AI assistant); explains which limit blocks a scenario that has no solution | optimizer, simulator, scenario builder, automation share, relaxed model |
| **Decision critic** | decision making | compares the options on value, service and risk, checks every rule of the brief, checks every number in the Writer's draft against the FactLedger, and pauses for the manager's approval | rule checks, FactLedger, policy comparison |
| **Writer** | generation | writes the answer for the reader: a chat reply, a one page memo for a country manager, or a note for finance | memo templates, tone by role |

The agents are coordinated as a **LangGraph** graph. The Planner is the entry node; the Analyst and the Optimizer run next, in parallel branches when the request names several countries; the Decision critic can send work back to the Optimizer when an option breaks a rule, and back to the Writer when a number cannot be traced; the graph stops for the manager's decision before anything is applied. The graph state keeps the request, the steps, every tool result and the model used at each step.

### Language models and routing

We will use three hosted models and one local model, all behind the same interface that StaffPilot already has:

| Model | Strength we measured or expect | Planned use |
|---|---|---|
| Claude Sonnet 5.5 (Anthropic, `claude-sonnet-5-5`) | measured: 100% intent accuracy on our 32 question holdout set | Planner, Decision critic: steps where a wrong reading is costly |
| GPT 5 nano (OpenAI, `gpt-5-nano`) | measured: 84% on the holdout, about 20 times cheaper per question than Claude | quick steps: classifying the request, filling scenario fields, short replies |
| Gemini 3 Flash (Google, `gemini-3-flash-preview`) | expected: long context and fluent writing at low cost; not yet measured, because our free Google key ran out of quota during the hackathon evaluation; the id is a preview name and will be checked against Google's model list in week 1 | Writer and Analyst: memos and explanations that read many tool results |
| Gemma 4 E4B (local, free, `google/gemma-4-e4b`) | measured: 88% on the holdout, runs on a laptop, no data leaves the machine | private mode, and the fallback when no API key is set |

**How the router chooses.** The router looks at three things before a step starts:

* the step type (classify, reason, decide, write);
* how hard the request is: how many countries it names, how many limits it sets, and whether those limits pull against each other;
* the user's setting: best quality, lowest cost or private.

It picks the cheapest model whose measured accuracy for that step type and difficulty is above a set level, using a table built from our evaluation. It **moves a step up** to a stronger model when the answer does not fit its schema or the FactLedger check fails. Every choice is shown in the UI with its reason, cost and time.

**How we will show that routing works.** We will extend our existing set of 82 labelled questions (main, follow up, edge and holdout sets) with multi step and multi country requests, and compare:

* every step on one model (each of the four models);
* the router.

We report task accuracy, the share of numbers traced to a tool, cost per request and time per request. The Evaluation page plots cost and time against accuracy for each setup, so it shows whether the router sits on the **Pareto frontier**: close to the accuracy of the most expensive model at a clearly lower cost.

### Advanced graph controls and reliability guardrails

* **Bounded critique loops.** The Decision critic has two ways to send work back: an option to the Optimizer (it breaks a rule) and a draft to the Writer (a number does not match the FactLedger). Each loop allows at most 2 retries, counted separately in the graph state, and LangGraph's `recursion_limit` is a hard stop for the whole run.
  * Optimizer loop: after the last retry the Critic passes on the best option with its open issues listed.
  * Writer loop: the second retry runs on a stronger model chosen by the router. If a number still does not match, the Writer leaves out the sentence that carries it and the answer says which figure could not be checked, so no number without a source reaches the user.
  * Two agents never pass work back and forth without end or run up the API bill.
* **Infeasible scenarios are explained, not reported as errors.** If a manager's limits leave the integer program with no solution in PuLP and CBC, the Optimizer solves a relaxed model where each hard limit (a hiring cap, a budget, a service goal) gets a penalised slack variable. It then reports which limit has to move and by how much, and the Critic weighs that trade off.
* **Every number is audited.** The Critic extracts every number from the Writer's draft and matches it against the FactLedger, the store of values produced by the tools, which StaffPilot already uses. A number that does not match, or that the model made up, sends the draft back to the Writer, within the retry limit above.
* **Real human in the loop.** With LangGraph checkpointing (an in memory saver during development, SQLite for the demo) and an interrupt after the Decision critic, the graph pauses, saves its state and shows the option cards. When the manager clicks "Ask for changes" and adds a limit (for example "keep hiring in Brazil to at most 5 people"), the new limit is written into the saved state and the run resumes from the Optimizer node, without repeating the earlier steps.
* **Parallel countries (map reduce).** For a request that names several countries, the Planner uses LangGraph's `Send` API to start one Analyst and Optimizer branch per country. A list reducer in the state merges their results before the Decision critic compares them. Adding a country adds a branch, not new code.
* **Two levels of offline use.** On a personal computer, Gemma 4 E4B runs through LM Studio for private or offline work. Automated tests and GitHub Actions use a deterministic backend instead (StaffPilot's offline router, plus scripted answers for the graph tests), so the full suite runs in seconds with no API key, no GPU and no model download.

### Interactive interface

A Gradio web app, extending the 11 page StaffPilot app:

* **Chat with live trace:** while the agents work, a side panel streams each step: which agent, which model and why, which tool, and the result.
* **Option cards:** the Decision critic's options side by side (value, advertisers served, wait, hires), each with an "Approve" or "Ask for changes" button that resumes the paused graph. Approved changes go to the decision log with the user's name.
* **Role view:** the same answer written for a manager, finance or a country manager.
* **Model settings:** choose best quality, lowest cost or private; see the cost of the session.
* **Evaluation page:** the router against each single model, with the Pareto chart.

### Scalability and adaptability

* Each agent is a node with its own prompt and tools; a new agent (for example a hiring market agent) is added by registering a node and an edge, without changing the others.
* Multi country requests fan out into parallel branches, so the graph grows with the request, not with the code.
* Models are listed in a config file; a new provider or model is one line, and the router learns its place from the evaluation table.
* A new year of data or a new country is added through the config, as in StaffPilot today.
* The tests and CI run with no API key and no local model, so anyone can run and grade the project for free.

### Repository

The final project lives in a new public repository, https://github.com/lequo-neu/staffpilot-crew. The hackathon repository, https://github.com/lequo-neu/staffpilot, stays exactly as it was submitted and is installed as a library, pinned to the submitted commit. The optimizer and simulator are reused without being copied, and everything in the new repository is final project work. Both repositories are public, so `pip install` needs no GitHub sign in; if the hackathon repository ever has to become private, we will commit a built wheel of it in the new repository instead.

### Tech stack

Python, LangGraph (checkpointing, interrupts, `Send`) and LangChain, Anthropic, OpenAI and Google APIs, LM Studio for local models, PuLP with CBC for the optimizer, pandas, Gradio, pytest and GitHub Actions.

## Demo features (end of week 4)

1. A manager types a new open ended request (for example the Brazil campaign) and watches the five agents plan, analyse, optimise, check and write the answer, with the model used at each step.
2. A request for three countries runs as three parallel branches and comes back as one comparison.
3. The Decision critic rejects an option that breaks a rule and sends it back; a scenario with no solution comes back with the limit that has to move.
4. The graph pauses on the option cards; the manager asks for a change and the run resumes from the Optimizer.
5. The same answer as a chat reply, a country memo and a finance note.
6. "What if an AI assistant onboards 30% of small advertisers in Brazil?": the human team the plan still needs, with the uplift assumption shown as a range.
7. A manager loads a month of real sign ups; the agents flag the gap with the forecast and propose next month's hires or notices for approval.
8. Switching between best quality, lowest cost and private mode, and the Evaluation page with the Pareto chart.

## Plan

| Week | Work | Lead |
|---|---|---|
| 1 | LangGraph graph with the five agents on the existing tools; bounded critique loops; router with fixed rules; deterministic test backend and CI; multi step questions written and labelled | Kevin (graph, router, CI), Khoa (questions) |
| 2 | Optimizer agent with alternatives, the automation share lever and the relaxed model; Decision critic with rule checks and the FactLedger audit; parallel country branches; Gemini added to the backends | Hamad (Optimizer, automation share), Kevin (critic, branches, Gemini) |
| 3 | Gradio chat with live trace; checkpointing and option cards that pause and resume the graph; monitoring of each monthly update; role views; router table from the evaluation | Hamad (UI, checkpointing), Khoa (Writer, role views, monitoring), Kevin (evaluation) |
| 4 | Full evaluation and Pareto chart, fixes, report, video and slides | all |

**Keeping the scope to four weeks.** Most of the work is the new agent layer; the planning engine is already built and tested (190 tests in the hackathon repository). The two features taken from the related products add little new code:

* the automation share lever reuses the hackathon's scenario builder and PuLP model: advertisers handled by an AI assistant leave the demand for human agents, which the model already handles as a change in sign ups, and their uplift is one extra value term;
* monthly monitoring reuses the hackathon's Monthly update page, forecast comparison and alerts; what is new is that a monthly file starts the graph by itself.

If time runs short, these two features are the first to be cut, and the core demo (features 1 to 5 and 8) does not depend on them.

## What is new compared with the hackathon

The hackathon delivered one assistant that answers questions about a plan. The final project adds what that assistant could not do: break a new request into steps, run countries in parallel, run and compare options, explain why a scenario has no solution, plan human and AI capacity together, watch each monthly update, reject its own bad options within a fixed number of retries, pause for the manager and resume from their change, write for different readers, and choose a model for each step with evidence that the choice saves cost without losing accuracy.
