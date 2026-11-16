# Final Project Proposal

**Team:** Team <number>
**Members:** Quoc Hung Le (Kevin), Hamad Al Sulaimi, Khoa Tran

## Project title

**StaffPilot Crew: a multi agent copilot for support staffing decisions**

## The problem

A company that supports advertisers in 22 countries must decide every month how many support agents to hire or release in each country. Sign ups change with the season and the market. Hiring takes a month to act, and too few agents means advertisers wait and leave. Today a staffing manager answers questions like these with spreadsheets and guesswork:

* "Sign ups in the USA came in 20% above forecast in March. What should we do, and what does it cost?"
* "We are thinking of launching a campaign in Brazil in Q3. Can our team handle it?"
* "Why does support in the UAE lose money, and what are our options?"
* "Write a short note to the Vietnam manager that explains why they are hiring."

In our Module 7 hackathon we built StaffPilot: an optimizer that plans the team for each country and month ($479.7M net value, $109.3M more than keeping the January team) and a single LangChain assistant that answers questions about that plan. The final project turns that single assistant into **a team of agents** that can take an open ended request, work out what needs to be done, run the analysis, weigh the options, and write the answer for the right person, while choosing **the best language model for each step**.

## Users

* **Staffing manager:** plans the year and reacts every month.
* **Country managers:** need their own numbers explained.
* **Finance:** needs proof that a change pays.
* **People team:** needs notices that are fair and planned early.

## Approach

### Agents and their roles

| Agent | Course role | What it does | Main tools |
|---|---|---|---|
| **Planner (supervisor)** | reasoning | reads the open ended request, splits it into steps, decides which agents to call and in what order, asks back when the request is unclear | task plan, conversation memory |
| **Analyst** | reasoning | explains the data: seasonality, forecast, which countries are short or over staffed, and why | forecast, data summary, RAG over the brief's rules and our assumptions |
| **Optimizer** | optimization | turns a request into a scenario and runs it: integer program per country plus the day by day simulator; tries alternatives (more hires, a service goal, a different policy) | optimizer, simulator, scenario builder |
| **Decision critic** | decision making | compares the options on value, service and risk, checks every rule of the brief, verifies that every number in the draft comes from a tool, and marks the step that needs a manager's approval | rule checks, number check (FactLedger), policy comparison |
| **Writer** | generation | writes the answer for the reader: a chat reply, a one page memo for a country manager, or a note for finance | memo templates, tone by role |

The agents are coordinated as a **LangGraph** graph. The Planner is the entry node; the Optimizer and the Analyst can run in parallel; the Decision critic can send work back to the Optimizer when an option breaks a rule or a number cannot be traced; the Writer is the last node. The graph state keeps the request, the steps, every tool result and the model used at each step.

### Language models and routing

We will use at least three hosted models and one local model, all behind the same interface that StaffPilot already has:

| Model | Strength we measured or expect | Planned use |
|---|---|---|
| Claude Sonnet 5.5 (Anthropic) | 100% intent accuracy on our 32 question holdout set | Planner, Decision critic: steps where a wrong reading is costly |
| GPT 5 nano (OpenAI) | 84% on the holdout, about 20 times cheaper per question than Claude | quick steps: classifying the request, filling scenario fields, short replies |
| Gemini (Google) | long context, strong writing | Writer and Analyst: memos and explanations that read many tool results |
| Gemma 4 E4B (local, free) | 88% on the holdout, runs offline, no data leaves the machine | private or offline mode, and a fallback when no API key is set |

**How the router chooses.** The router sees the step type (classify, reason, decide, write), the length of the context and the user's setting (best quality, lowest cost or private). It picks the cheapest model whose measured accuracy for that step type is above a set level, using a table built from our evaluation. It moves a step up to a stronger model when the number check fails or the cheap model's answer does not parse. Every choice is shown in the UI with its reason, cost and time.

**How we will show that routing works.** We will extend our existing set of 82 labelled questions (main, follow up, edge and holdout sets) with multi step requests, and compare:

* every step on one model (each of the four models);
* the router.

We report task accuracy, numbers traced to a tool, cost per request and time per request. The goal is to match the best single model's accuracy at a clearly lower cost.

### Interactive interface

A Gradio web app, extending the 11 page StaffPilot app:

* **Chat with live trace:** while the agents work, a side panel streams each step: which agent, which model and why, which tool, and the result.
* **Option cards:** the Decision critic's options side by side (value, advertisers served, wait, hires), each with an "Approve" or "Ask for changes" button. Approved changes go to the decision log with the user's name.
* **Role view:** the same answer written for a manager, finance or a country manager.
* **Model settings:** choose best quality, lowest cost or private; see the cost of the session.

### Scalability and adaptability

* Each agent is a node with its own prompt and tools; a new agent (for example a hiring market agent) is added by registering a node and an edge, without changing the others.
* Models are listed in a config file; a new provider or model is one line, and the router learns its place from the evaluation table.
* A new year of data or a new country is added through the config, as in StaffPilot today.
* The app, tests and CI keep working offline with no API key, so anyone can run and grade it for free.

### Repository

The final project lives in a new repository, `staffpilot-crew`. The hackathon repository stays exactly as it was submitted and is installed as a library, pinned to the submitted commit, so the optimizer and simulator are reused without being copied and everything in the new repository is final project work.

### Tech stack

Python, LangGraph and LangChain, Anthropic, OpenAI and Google APIs, LM Studio for local models, PuLP with CBC for the optimizer, pandas, Gradio, pytest and GitHub Actions.

## Demo features (end of week 4)

1. A manager types a new open ended request (for example the Brazil campaign) and watches the five agents plan, analyse, optimise, check and write the answer, with the model used at each step.
2. The Decision critic rejects an option that breaks a rule and sends it back; the Optimizer returns a fixed option.
3. The same answer as a chat reply, a country memo and a finance note.
4. Switching between best quality, lowest cost and private mode, with the change in cost and accuracy shown.
5. An evaluation page: router against each single model on accuracy, cost and time.

## Plan

| Week | Work | Lead |
|---|---|---|
| 1 | LangGraph graph with the five agents on the existing tools; router with fixed rules; multi step questions written and labelled | Kevin (graph, router), Khoa (questions) |
| 2 | Optimizer agent with alternatives; Decision critic with rule and number checks; Gemini added to the backends | Hamad (Optimizer), Kevin (critic, Gemini) |
| 3 | Gradio chat with live trace, option cards with approval, role views; router table from the evaluation | Hamad (UI), Khoa (Writer, role views), Kevin (evaluation) |
| 4 | Full evaluation, fixes, report, video and slides | all |

## What is new compared with the hackathon

The hackathon delivered one assistant that answers questions about a plan. The final project adds what that assistant could not do: break a new request into steps, run and compare options, reject its own bad options, write for different readers, and choose a model for each step with evidence that the choice saves cost without losing accuracy.
