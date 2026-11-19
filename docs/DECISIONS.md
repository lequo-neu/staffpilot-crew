# Decisions made while building

Where the code differs from the proposal, or where the proposal left a choice open, the choice and its reason are recorded here.

## Models

* **Gemini 3.5 Flash instead of Gemini 3 Flash.** The proposal named `gemini-3-flash-preview`. The hackathon's own online configuration (`config/eval_online.toml` of StaffPilot) pins `google_genai:gemini-3.5-flash`, the Google model the team's key can call. The crew uses the pinned id so both projects call the same model; it can be changed in `config/crew.toml`.
* **Prices from the hackathon only.** The prices in `default_crew.toml` are the ones the hackathon looked up on 3 November 2026 (Claude Sonnet 5.5 $2 and $10, GPT 5 nano $0.05 and $0.40 per million tokens). No price was looked up for Gemini 3.5 Flash, so its cost is shown as unknown rather than guessed.
* **Priors are labelled.** Accuracies for steps the hackathon did not measure are prior guesses. Every routing record says whether its accuracy was measured or a prior, and `eval --online` replaces priors with measured values.
* **Escalation picks the assigned model on a tie.** When two models have the same accuracy on a step, the Writer's escalation goes to the model the proposal assigned to that step, then to the cheaper one.

## Planning

* **The hackathon planner is used through one module.** `planning.py` is the only place that calls the planner. Options are run with `Planner.run` on the context built by `Planner._scenario_context`, because `Planner.scenario` returns totals without waiting times. The method is private, but the package is pinned to the submitted commit, so it cannot change under the crew.
* **Feasibility is decided by the crew.** The hackathon's optimizer treats the service goal as a soft floor and has no budget, so it never returns "no solution". The crew calls a plan infeasible when it misses the goal by more than half a point, costs more than the budget or hires more than the cap in a month.
* **The relaxed model is a separate linear program.** Adding slack variables inside the hackathon's integer program would mean changing that repository, which the team decided not to do. The crew solves a monthly capacity model of the same year with PuLP and CBC, one slack variable per limit with a penalty, once per limit. Each estimate is then checked with the real optimizer and raised step by step when the check fails, so the numbers shown are always the planner's.
* **Automation share is a lower demand.** Advertisers handled by an AI assistant leave the demand for human agents (a multiplier on sign ups in every month, which the hackathon already supports). Their uplift is unknown, so it is reported as a range set in the config.
* **The hackathon router misreads automation.** Its offline router reads "an AI assistant handles 30%" as 30 advertisers per agent, as a 30% change in sign ups, or as a question back. The Planner drops those readings when the crew has read the same number as an automation share.

## Graph

* **Two version counters for country results.** Each Optimizer result carries a `version` that grows with every run of the thread (so a resumed run always replaces older results through the reducer) and an `attempt` (the critic's retry number, which decides which alternative to try).
* **The critic's number check is its own node.** The proposal gives the Decision critic two jobs, comparing options and checking numbers. They are two nodes (`critic` and `number_check`) with the same agent name in the trace, so each loop has its own edge and counter.
* **Sentences, not answers, are dropped.** After the Writer's last retry, only the sentences that still hold a number without a source are left out, and the answer says how many.
* **Role switching does not call a model.** Switching the reader after a run rewrites the answer from the ledger with the Writer's templates, so it is instant and free; a new request for another reader goes through the Writer as usual.

## Interface and operations

* **The hackathon pages are mounted, not copied.** The crew app serves `staffpilot.app.build_app` at `/planner/` on the same planner, so those pages run exactly as submitted.
* **CI starts only by hand.** The team's GitHub Actions budget ran out on 16 November 2026, so the workflow keeps only `workflow_dispatch` until the team turns automatic runs back on.
* **Sign in is the crew's own.** The hackathon app signs in through Gradio's `auth` and a JSON file. The crew needs roles, a country per user and the 90 day rule, so it keeps users in SQLite (`auth.py`) with PBKDF2 from the standard library, and shows its own form. The hackathon pages at /planner/ are mounted without the crew's sign in.
* **Users and decisions share one SQLite file.** `outputs/crew_app.sqlite` holds the `users` and `decision_log` tables; the decision log moved there from a CSV file so each row can carry the approver's username, name and role. LangGraph's checkpoints stay in their own file, which LangGraph manages.
* **Who may approve.** Only a staffing manager (and the admin) gives the final approval. Finance first approved "budget options"; with the proposal workflow it confirms budgets instead (the Pending Finance step), so there is one final approver. The rule is checked in the app, in the workflow and again inside the graph's approval node.
* **People Team is a fourth Writer view.** The hackathon had three readers. The People Team view lists the hires and notices decided in each month (now kept on every option), the one month lead times and a reminder to plan the communication before notices.
* **A country manager's country is a fallback.** It is used only when a request names no country, so a country manager can still ask about another one.
* **An administrator with a short password.** The team asked for a user `Admin` with the password `Admin` and full rights. Usernames are case insensitive, so `Admin` and `admin` both work. The password breaks the 8 character rule, so it is allowed only for this seeded account; each sign in with it shows a banner asking for a new one, and a new one must follow the rules. The admin may approve every option and sees a Users tab. Databases made before the admin role existed are rebuilt with the wider role check, keeping their users.
* **Demo accounts in the README, not on the sign in page.** They are created only when the users table is empty, and one is seeded 120 days old so the warning shows at once. The sign in page lists no accounts, as a real one would not.
* **The pages follow the user's task, not the system.** After two review rounds in the browser, the first page is a question box with the common questions for the user's role, then the recommendation, the option cards and the decision. The agents' steps, the model settings and the model cost are folded away; a step names its model only when a model wrote it. Two options that lead to the same plan show as one card. Mistakes such as an empty question show a short notice instead of marking the page red. The tabs are named for what they are for: Decide, Monthly check, Decisions, Full planner, Answer quality.

## The daily desk

* **Proposals wrap crew runs.** A proposal keeps a snapshot of the run it came from (the cards, the chosen option, the plan the year started with and the month by month schedules), so its numbers can be read and exported later even if the run is gone. Approving a proposal also resumes its paused LangGraph run, so the decision log keeps working as before.
* **Finance only when it matters.** Sending every change to Finance would make it a bottleneck. A proposal goes to Finance when its staffing cost moves more than 5% or $100,000, when its hires change by more than 5 in a country, or when it sets a budget; the thresholds are in the config. New numbers that cross a threshold send a proposal back to Finance.
* **Who acts on what.** Drafts, returned proposals, withdrawing and revising belong to the creator; Finance confirms or returns; the staffing manager approves or returns; the People Team starts hiring and marks it done. A country manager may propose for their own country only. Reviewers read everything and change nothing.
* **Notes are required for every action,** including creating a draft, because the next person reads them on the timeline instead of asking.
* **Out of date means locked.** A proposal is out of date when its country had a monthly check after its numbers were made, when the dataset changed, or after 7 days. Approval is locked, not hidden, so the approver sees why and refreshes. Refresh runs the request again with the same limits and keeps the chosen option when it still exists.
* **Deadlines are dates, not countdowns.** Agents needed from month M must be decided one lead time earlier, so the deadline is the first day of the decision month in the planning year. With the course data (planning 2026, today in November 2026) most of them are overdue, which is the truth about that plan.
* **Today is computed when it is opened,** from the proposals, the plan's alerts and the monthly checks; there is no background job, because new data only arrives with a monthly check.
* **The sidebar is the assistant.** Questions run in Gradio's `Sidebar` on the right, so Today and Proposals stay in view; a run with options moves the page to Workspace.
* **The reviewer account is not shown on the sign in page.** The request to list a reviewer or admin account there came after the decision to show no accounts on that page, so it is listed in the README with the others.
* **Out of scope:** email and Slack, scheduled jobs and an HR system connection. The email text can be copied, the hiring plan exported as CSV, and each handoff appears on the next person's Today page.
