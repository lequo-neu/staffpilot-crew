# How AI tools were used

The course allows AI tools if their use is explained. This page says which parts were drafted with AI and how the team checks them. It will be completed as the team reviews the work.

## What was AI assisted

| Part | AI role | How it is checked |
|---|---|---|
| Proposal and related products (`docs/PROPOSAL.md`, `docs/RELATED_PRODUCTS.md`) | drafted by a coding assistant (Claude Code) from the team's hackathon and the vendors' public pages | Quoc Hung Le edited them; review points from the team were applied; the vendors' claims are marked as their own |
| First version of the code in `src/staffpilot_crew/` and `tests/` (16 November 2026) | written by Claude Code from the proposal, on Quoc Hung Le's request, reusing the hackathon package | 57 offline tests, lint, runs on the synthetic sample and on the course data, and a check of the app in the browser; review by Hamad Al Sulaimi and Khoa Tran is still to come |
| Sign in, roles and SQLite storage (`auth.py`, changes in `app.py`, `graph.py`, `services.py`, `agents.py`), 16 November 2026 | written by Claude Code from Quoc Hung Le's specification | 20 more offline tests (77 in all), and a check in the browser: sign in, the 90 day banner, a password change, a country manager's locked Approve and a staffing manager's approval in the SQLite log |
| The daily desk: proposals, timeline, Today, sidebar, deltas, out of date lock, saved scenarios, CSV, copy, role based panels (`workflow.py`, `desk.py`, `app.py`), 16 to 17 November 2026 | written by Claude Code from Quoc Hung Le's P1 and P2 specification | 98 offline tests, and a handoff checked in the browser: a country manager's proposal, a staffing manager's approval with its timeline, the People Team's hiring step and CSV, and the reviewer's technical panels |
| `README.md`, `docs/DECISIONS.md` | drafted by Claude Code | every number in them was checked against a test or a run |

## What the team decided

* The topic, the five agents and their roles, the models and the four week plan (the proposal).
* To keep the hackathon repository exactly as submitted and use it as a pinned library.
* To keep the tests and CI free of paid calls, and to run CI only by hand while the Actions budget is used up.

## What was not used

No code was copied from other teams or from other courses.
