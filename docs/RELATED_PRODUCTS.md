# Related products

Before fixing the scope we looked at commercial tools that already put AI agents into staffing decisions, to learn what users of such tools expect and to choose, from what we already have, what is worth adding in four weeks. Everything below about these products comes from the vendors' own pages and press releases (read on 16 November 2026); we have not used either product.

## The two products we compared

We looked at several tools (NiCE, Anaplan, Assembled, Calabrio, Microsoft Dynamics 365 workforce engagement) and kept two that stand at the two ends of our problem:

* **NiCE Workforce Management with its copilots** (formerly CXone Mpower WFM): the daily operations end, for contact centers.
* **Anaplan role based agents, especially the Workforce Analyst**: the planning end, for headcount and scenarios across a business.

StaffPilot sits between them: monthly hiring and notices for a year, by country.

### NiCE Workforce Management and its copilots

What the vendor states:

* AI driven forecasting and scheduling across all channels that "plans capacity across human and AI agents", with intraday adjustments ([NiCE](https://www.nice.com/products/copilots)).
* A Workforce Empowerment Suite, launched in June 2026, that joins workforce management, quality, performance and AI operations on one platform, with a Copilot for Workforce Managers that gives one view of human and AI agents ([UC Today](https://www.uctoday.com/workplace-management/nice-launches-workforce-empowerment-suite-to-manage-the-hybrid-human-ai-contact-center/)).
* Scheduling with machine learning, shift bidding and time off requests in a mobile app, and discrete event simulation of digital channels ([NiCE datasheet](https://resources.nice.com/wp-content/uploads/2026/04/0536831-Datasheet-CXone-Workforce-Management.pdf)).
* Copilots for agents and supervisors; the page is not consistent on whether workforce managers have their own copilot ([NiCE](https://www.nice.com/products/copilots)).

What we did not find: a description of what if scenarios, of how a recommendation is explained, of approval before a change, or of which language models are used.

### Anaplan role based agents (Workforce Analyst)

What the vendor states ([Anaplan press release, 9 December 2025](https://www.anaplan.com/news/anaplan-introduces-new-suite-role-based-ai-agents/)):

* A suite of agents, each built for one business role: Finance Analyst, Supply Chain Analyst, Sales Analyst, Workforce Analyst, and CoModeler, which builds planning models from plain language and documents every step it takes.
* The Workforce Analyst identifies workforce risks, works out and communicates the impact of headcount decisions, and answers questions in real time.
* Agents analyse data, write reports, recommend actions and carry out **approved** changes, such as moving resources after a scenario or publishing updated plans to stakeholders. Autonomous agents are announced "always with human oversight".
* For contact centers, monitoring agents surface risks and recommend actions as conditions change; planners ask what if questions such as the best mix of human agents, new hires and automation for the next 18 months ("buy, build or bot") ([Anaplan blog](https://www.anaplan.com/blog/ai-for-contact-center-planning-faster-scenario-modeling/)).

What we did not find: which models are used, or how a number in an answer is traced to its source; the press release mentions explainability only for its forecasting engine.

## How they compare with StaffPilot Crew

| Need | NiCE | Anaplan | StaffPilot today (hackathon) | StaffPilot Crew (planned) |
|---|---|---|---|---|
| Horizon | intraday to weeks | months to years | a year, month by month | the same |
| Human and AI capacity in one plan | yes | yes ("buy, build or bot") | no, human agents only | **adopted:** an automation share lever |
| What if scenarios in plain words | not described | yes | yes, one at a time | many steps, several countries in parallel |
| Agents split by role | copilots by user role | one agent per business role | one assistant | five agents by task, answers written for each user role |
| Watches new data and raises risks | intraday adjustments | monitoring agents | alerts on the This year page | **adopted:** the Analyst reviews every monthly update by itself |
| Acts only after approval | not described | yes, approved changes | decision log with the user's name | graph pauses and resumes after approval |
| Shows how it reached a result | not described | CoModeler documents its steps | the question as read, numbers matched to planner facts | live trace of every agent, model and tool |
| Every number traced to a tool | not described | not described | yes (FactLedger) | yes, and a failed check sends the draft back |
| Choice of language model | not stated | not stated | any provider, or none | a router with measured accuracy and cost per step |
| Runs without a paid key | no | no | yes | yes |
| Optimizer the user can inspect | not described | not described | an integer program and a simulator in the repository | the same, plus why a scenario has no solution |

## What we take from them

We only add what fits our data (monthly sign ups, salaries, existing teams) and our four weeks.

1. **An automation share lever** (from NiCE's human plus AI capacity and Anaplan's "buy, build or bot"). A manager can ask "what if an AI assistant handles the onboarding of 30% of small advertisers in Brazil?". The Optimizer treats that share as advertisers who need no human agent and plans the human team for the rest. Because the brief gives no data on how much uplift an AI assistant earns, its uplift is a parameter (zero by default) and the result is reported as a range in the assumptions, not as a fact.
2. **Monitoring on each monthly update** (from Anaplan's monitoring agents and NiCE's adjustments as conditions change). When a manager loads a month of real sign ups, the Planner starts a run without being asked: the Analyst compares it with the forecast, and the Decision critic proposes the hires or notices for the next month as option cards that wait for approval.
3. **Approval before action, written into the design** (both products). This confirms our choice to pause the graph before anything is applied and to log who approved what.
4. **A record of every step** (CoModeler). This confirms the live trace, and we will also save it with each decision in the log.

## What we leave out, and why

* **Intraday scheduling, shifts, breaks and time off** (NiCE): the course data are monthly and the brief decides hires and notices by month.
* **Quality review of every interaction** (NiCE): we have no conversation data.
* **Building planning models from plain language** (CoModeler): our model is fixed by the brief; changing it from a chat would make the results impossible to check.
* **Agents for finance, sales and supply chain** (Anaplan): outside the brief.

## What we keep as our own

Neither vendor states which models it uses or how a number in an answer is traced. Our project makes those two things its main result: a router whose choices are measured on our own labelled questions, and a check that every number in an answer comes from a tool. It also stays open and free to run: the optimizer is in the repository and the tests need no key.

## Sources

* NiCE, AI Workforce Empowerment and copilots: https://www.nice.com/products/copilots
* NiCE Workforce Management datasheet (April 2026): https://resources.nice.com/wp-content/uploads/2026/04/0536831-Datasheet-CXone-Workforce-Management.pdf
* UC Today, NICE launches Workforce Empowerment Suite (June 2026): https://www.uctoday.com/workplace-management/nice-launches-workforce-empowerment-suite-to-manage-the-hybrid-human-ai-contact-center/
* Anaplan, role based AI agents (9 December 2025): https://www.anaplan.com/news/anaplan-introduces-new-suite-role-based-ai-agents/
* Anaplan, intelligent role based agents: https://www.anaplan.com/platform/intelligent-role-based-agents/
* Anaplan, AI for contact center planning: https://www.anaplan.com/blog/ai-for-contact-center-planning-faster-scenario-modeling/
* Assembled, support orchestration suite (also considered): https://www.assembled.com/blog/balance-ai-agents-with-human-workforce-management
