"""The five agents. Each is a plain function of the shared services and the graph state.

Numbers never come from a model. The Analyst and the Optimizer put every figure into the
FactLedger; models (when one is reachable) only write prose with ``{{placeholders}}`` that the
ledger fills in, and the Decision critic rejects any number the ledger does not hold. Without
a model every agent still works: the deterministic offline backend reads the request and the
templates below write the answer.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
from staffpilot.agent.facts import FactLedger
from staffpilot.agent.schemas import Intent, RoutedQuery
from staffpilot.agent.tools import ToolOutput
from staffpilot.planner import RECOMMENDED_POLICY
from staffpilot.scenarios import ScenarioParams

from staffpilot_crew import planning
from staffpilot_crew.constraints import AUDIENCE_LABELS, Constraints, read_audience, read_constraints
from staffpilot_crew.models import OFFLINE
from staffpilot_crew.relax import find_relaxations, meets
from staffpilot_crew.router import Difficulty, RouteChoice, assess
from staffpilot_crew.services import CrewServices
from staffpilot_crew.state import CountryTask, CrewState


MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
ALL_COUNTRIES = re.compile(r"\b(all|every|each)\s+(countries|country|markets?)\b|\bcompany\b|\boverall\b|\bglobal(ly)?\b", re.IGNORECASE)


def key(country: str) -> str:
    return country.replace(" ", "_")


def event(agent: str, action: str, detail: str = "", route: RouteChoice | None = None, seconds: float = 0.0, **extra: Any) -> dict[str, Any]:
    record = {"agent": agent, "action": action, "detail": detail, "seconds": round(seconds, 3), **extra}
    if route is not None:
        record.update(model=route.label, model_name=route.model, reason=route.reason, accuracy_source=route.accuracy_source)
    return record


def difficulty_of(state: CrewState | CountryTask) -> Difficulty:
    data = (state.get("plan") or {}).get("difficulty") or {}
    return Difficulty(data.get("countries", 1), data.get("constraints", 0), data.get("conflicts", []), data.get("level", "simple"))


def narrate(
    services: CrewServices,
    agent: str,
    step: str,
    question: str,
    facts: dict[str, float],
    notes: list[str],
    difficulty: Difficulty,
    mode: str,
    countries: list[str],
    route: RoutedQuery | None = None,
    feedback: str = "",
    choice: RouteChoice | None = None,
) -> tuple[str | None, RouteChoice, dict[str, Any] | None, list[str]]:
    """Prose from the routed model, rendered and checked; None when there is no model or the check fails.

    Returns the text, the route, the call record and the numbers the check rejected.
    """
    choice = choice or services.router().choose(step, difficulty, mode)
    if choice.model == OFFLINE:
        return None, choice, None, []
    client = services.registry.client(choice.model)
    ledger = FactLedger(dict(facts))
    output = ToolOutput(f"{agent.lower().replace(' ', '_')}_facts", dict(facts), notes=notes)
    reading = route or RoutedQuery(intent=Intent.plan_lookup, countries=countries)
    text, record = client.call("compose", question, reading, [output], ledger.formatted(), feedback, agent=agent, step=step)
    if not record.ok or not text:
        return None, choice, record.as_dict(), []
    rendered, missing = ledger.render(text)
    unsupported = missing + ledger.verify(rendered, context=question + " " + " ".join(notes), countries=countries)
    return (rendered if not unsupported else None), choice, record.as_dict(), unsupported


# Planner (supervisor)


def without_crew_numbers(reading: RoutedQuery, limits: Constraints) -> RoutedQuery:
    """Drop what the hackathon router read from a number the crew already read as a limit.

    That router does not know about an AI assistant's share, so "an AI assistant handles 30%"
    can come back as 30 advertisers per agent, a 30% change in sign ups, or a question back.
    The crew's own reading of the share is the one kept.
    """
    if not limits.automation_share:
        return reading
    share = limits.automation_share
    update: dict[str, Any] = {"clarification": None}
    if reading.intent == Intent.out_of_scope:
        update["intent"] = Intent.what_if
    if reading.scenario is not None:
        scenario = reading.scenario
        changes: dict[str, Any] = {}
        if scenario.capacity_per_agent == round(share * 100):
            changes["capacity_per_agent"] = None
        if any(abs(scenario.demand_multiplier - m) < 1e-9 for m in (1 - share, 1 + share)):
            changes["demand_multiplier"] = 1.0
        scenario = scenario.model_copy(update=changes)
        unchanged = ScenarioParams(countries=scenario.countries, months=scenario.months)
        update["scenario"] = None if scenario == unchanged else scenario
    return reading.model_copy(update=update)



def planner_node(services: CrewServices, state: CrewState) -> dict[str, Any]:
    started = time.perf_counter()
    planner = services.planner
    request = state.get("request", "")
    mode = state.get("mode", "best_quality")
    limits = read_constraints(request).merged(Constraints.model_validate(state.get("user_constraints") or {}))
    audience = state.get("audience") or read_audience(request)
    trace, calls, routing = [], [], []

    if state.get("trigger") == "monthly_update":
        monthly = state["monthly"]
        country = planner.resolve_country(monthly["country"]) or monthly["country"]
        difficulty = assess(1, limits.count(), limits.conflicts(), services.crew.router.hard_countries)
        plan = {
            "intent": "monthly_update",
            "countries": [country],
            "scenario": None,
            "difficulty": difficulty.__dict__,
            "steps": [
                f"Analyst: compare {country}'s sign ups so far with the forecast",
                f"Optimizer: plan {country} again from {MONTHS[int(monthly['as_of']) - 1]}",
                "Decision critic: check the rules and propose next month's change",
                f"Writer: write the update for the {AUDIENCE_LABELS[audience].lower()}",
            ],
            "read_by": "monthly update file",
        }
        trace.append(event("Planner", "monthly update received", f"{country}, as of {MONTHS[int(monthly['as_of']) - 1]}", seconds=time.perf_counter() - started))
        return {"plan": plan, "user_constraints": limits.model_dump(), "audience": audience, "approval_status": "Planning", "trace": trace, "retry_counts": {"optimizer": 0, "writer": 0}}

    # A first reading with the deterministic router gives the countries, so the difficulty is
    # known before a model is chosen for the full reading.
    offline = services.registry.client(OFFLINE)
    first, _ = offline.call("route", request, planner.countries, agent="Planner", step="reason")
    countries = [c for c in (planner.resolve_country(n) for n in first.countries) if c]
    if not countries and ALL_COUNTRIES.search(request):
        countries = list(planner.countries)
    # A country manager's own country stands in when the request names no country.
    home = planner.resolve_country(state["home_country"]) if state.get("home_country") else None
    used_home = False
    if not countries and home:
        countries, used_home = [home], True
    difficulty = assess(len(countries), limits.count(), limits.conflicts(), services.crew.router.hard_countries)
    choice = services.router().choose("reason", difficulty, mode)
    reading, read_by = first, "deterministic offline router"
    if choice.model != OFFLINE:
        result, record = services.registry.client(choice.model).call("route", request, planner.countries, agent="Planner", step="reason")
        calls.append(record.as_dict())
        if record.ok and result is not None:
            reading, read_by = result, choice.label
            named = [c for c in (planner.resolve_country(n) for n in reading.countries) if c]
            countries = named or countries
            used_home = used_home and not named
        else:
            trace.append(event("Planner", "model call failed, kept the offline reading", record.error, choice))
    routing.append({"agent": "Planner", **choice.as_dict()})
    reading = without_crew_numbers(reading, limits)
    if reading.clarification and reading.scenario is None and limits.count():
        # The crew read the manager's limits itself, so there is nothing left to ask.
        reading = reading.model_copy(update={"clarification": None})

    clarification = None
    if reading.intent == Intent.out_of_scope and not limits.automation_share:
        clarification = "I can help with staffing plans, scenarios and decisions for the countries in the data. What would you like to plan?"
    elif reading.clarification:
        clarification = reading.clarification
    elif not countries:
        sample = ", ".join(planner.countries[:3])
        clarification = f"Which countries should I look at? For example: {sample}, or say all countries."
    if clarification:
        trace.append(event("Planner", "asked back", clarification, choice, time.perf_counter() - started))
        return {
            "plan": {"intent": str(reading.intent), "countries": [], "clarification": clarification, "difficulty": difficulty.__dict__},
            "answer": clarification,
            "approval_status": "Needs details",
            "trace": trace,
            "routing_trace": routing,
            "calls": calls,
        }

    difficulty = assess(len(countries), limits.count(), limits.conflicts(), services.crew.router.hard_countries)
    scenario = reading.scenario.model_dump() if reading.scenario is not None else None
    plan = {
        "intent": str(reading.intent),
        "countries": countries,
        "months": list(reading.months),
        "scenario": scenario,
        "difficulty": difficulty.__dict__,
        "steps": [
            f"Analyst: review demand and the current team in {', '.join(countries)}",
            "Optimizer: run the recommended plan and the alternatives"
            + (" within your limits" if limits.count() else ""),
            "Decision critic: check every rule and compare the options",
            f"Writer: answer for the {AUDIENCE_LABELS[audience].lower()}",
        ],
        "read_by": read_by,
        "home_country_used": used_home,
    }
    detail = f"{len(countries)} {'country' if len(countries) == 1 else 'countries'}; difficulty {difficulty.describe()}"
    if used_home:
        detail += f"; no country named, so your assigned country ({home}) is used"
    if limits.describe():
        detail += "; limits: " + ", ".join(limits.describe())
    trace.append(event("Planner", "planned the steps", detail, choice, time.perf_counter() - started, tool="request reading"))
    return {
        "plan": plan,
        "user_constraints": limits.model_dump(),
        "audience": audience,
        "approval_status": "Planning",
        "retry_counts": {"optimizer": 0, "writer": 0},
        "trace": trace,
        "routing_trace": routing,
        "calls": calls,
    }


# Analyst (one branch per country)


def analyst_node(services: CrewServices, task: CountryTask) -> dict[str, Any]:
    started = time.perf_counter()
    planner = services.planner
    country = task["country"]
    k = key(country)
    base = planner.run(RECOMMENDED_POLICY).outcomes[country].kpis
    demand = np.asarray(planner.forecast.demand(country)[:12], dtype=float)
    peak, low = int(np.argmax(demand)), int(np.argmin(demand))
    facts = {
        "start_agents": float(base["start_agents"]),
        "needed_agents": float(base["avg_active"]),
        "peak_demand": float(demand[peak]),
        "low_demand": float(demand[low]),
        "yearly_demand": float(demand.sum()),
    }
    findings = [
        (f"Eligible sign ups in {country} peak in {MONTHS[peak]} at {{{{{k}.analysis.peak_demand}}}} and are lowest in "
        f"{MONTHS[low]} at {{{{{k}.analysis.low_demand}}}}.")
    ]
    start, need = facts["start_agents"], facts["needed_agents"]
    if need > start * 1.1:
        findings.append(f"The team of {{{{{k}.analysis.start_agents}}}} agents is short: the plan needs about {{{{{k}.analysis.needed_agents}}}} on average.")
    elif start > need * 1.1:
        findings.append(f"The team of {{{{{k}.analysis.start_agents}}}} agents is larger than the {{{{{k}.analysis.needed_agents}}}} the plan needs on average.")
    else:
        findings.append(f"The team of {{{{{k}.analysis.start_agents}}}} agents is close to what the plan needs.")
    if task.get("trigger") == "monthly_update":
        monthly = task["monthly"]
        done = int(monthly["as_of"]) - 1
        actual = float(np.sum(monthly["actual_signups"]))
        expected = float(demand[:done].sum())
        facts.update({"actual_signups": actual, "forecast_signups": expected, "signup_change": actual / expected - 1 if expected else 0.0})
        findings.append(
            f"Sign ups so far were {{{{{k}.analysis.actual_signups}}}} against a forecast of {{{{{k}.analysis.forecast_signups}}}} "
            f"({{{{{k}.analysis.signup_change}}}} apart)."
        )
    passages = services.retriever.search(task.get("request") or country, k=2)
    sources = [p.citation for p in passages]
    ledger_facts = {f"{k}.analysis.{name}": value for name, value in facts.items()}
    analysis = {"country": country, "findings": findings, "sources": sources, "narrative": None}

    calls, routing = [], []
    text, choice, record, rejected = narrate(
        services,
        "Analyst",
        "write",
        f"In two sentences, explain demand and the current team in {country}. Use only the facts given.",
        ledger_facts,
        [p.text[:400] for p in passages],
        difficulty_of(task),
        task.get("mode", "best_quality"),
        [country],
    )
    routing.append({"agent": "Analyst", "country": country, **choice.as_dict()})
    if record:
        calls.append(record)
    analysis["narrative"] = text
    detail = f"{country}: {len(findings)} findings, {len(sources)} rule passages"
    if rejected:
        detail += f"; the model's text was dropped ({len(rejected)} numbers without a source)"
    trace = [event("Analyst", "reviewed the data", detail, choice, time.perf_counter() - started, tool="forecast, plan, rule search", country=country)]
    return {"analyses": {country: analysis}, "fact_ledger": ledger_facts, "trace": trace, "routing_trace": routing, "calls": calls}


# Optimizer (one branch per country)


def optimizer_node(services: CrewServices, task: CountryTask) -> dict[str, Any]:
    started = time.perf_counter()
    planner = services.planner
    country = task["country"]
    k = key(country)
    limits = Constraints.model_validate(task.get("user_constraints") or {})
    plan = task.get("plan") or {}
    attempt = int(task.get("attempt", 0))
    version = int(task.get("version", 0))
    feedback = task.get("feedback") or []
    automation = (services.crew.automation.uplift_share_low, services.crew.automation.uplift_share_high)
    options: list[dict[str, Any]] = []
    strict_issues: list[str] = []
    relaxations: list[dict[str, Any]] = []
    tools: list[str] = []

    base_outcome = planner.run(RECOMMENDED_POLICY).outcomes[country]
    base_kpis = {name: float(v) for name, v in base_outcome.kpis.items() if name in planning.KPI_KEYS}
    facts = {f"{k}.base.{name}": value for name, value in base_kpis.items()}
    rules = services.settings.rules
    facts.update({"rules.ramp_months": float(rules.ramp_months), "rules.notice_months": float(rules.notice_months)})

    if task.get("trigger") == "monthly_update":
        monthly = task["monthly"]
        update = planner.monthly_update(country, int(monthly["as_of"]), list(monthly["actual_signups"]), int(monthly.get("headcount_change", 0)))
        for name, outcome in (("update", update.updated), ("keep", update.kept)):
            kpis = {key_: float(outcome.kpis.get(key_, 0.0)) for key_ in planning.KPI_KEYS}
            peak = float(np.max(outcome.plan.hires[:12]))
            options.append(planning.option(country, "recommended" if name == "update" else "profit_only", kpis, peak, [], limits, automation, months=planning.schedule(outcome.plan)))
            options[-1].update(name=name, label="Updated plan" if name == "update" else "Keep the approved plan")
        month = int(monthly["as_of"]) - 1
        next_month = {
            "month": MONTHS[month],
            "hires": float(update.updated.plan.hires[month]),
            "notices": float(update.updated.plan.notices[month]),
            "planned_hires": float(update.original.hires[month]),
            "planned_notices": float(update.original.notices[month]),
        }
        facts.update({f"{k}.next.{name}": value for name, value in next_month.items() if name != "month"})
        facts.update({f"{k}.monthly.{name}": float(value) for name, value in update.facts().items()})
        tools.append("monthly update (optimizer and simulator)")
    else:
        free = Constraints(automation_share=limits.automation_share)
        params = planning.country_params(plan.get("scenario"), country, free)
        for name, policy in (("recommended", RECOMMENDED_POLICY), ("profit_only", "optimized")):
            kpis, peak, notes, months = planning.evaluate_full(planner, params, policy)
            options.append(planning.option(country, name, kpis, peak, notes, limits, automation, months=months))
        tools.append("optimizer and simulator")
        if limits.count():
            limited = planning.country_params(plan.get("scenario"), country, limits)
            kpis, peak, notes, months = planning.evaluate_full(planner, limited, "optimized")
            options.append(planning.option(country, "within_limits", kpis, peak, notes, limits, automation, months=months))
            strict_issues = meets(kpis, limits, peak)
            if strict_issues:
                def check(trial: Constraints) -> tuple[dict[str, float], float]:
                    result = planning.evaluate(planner, planning.country_params(plan.get("scenario"), country, trial), "optimized")
                    return result[0], result[1]

                found = find_relaxations(planning.year_inputs(planner, limited), limits, check)
                relaxations = [r.as_dict() for r in found]
                tools.append("relaxed model (PuLP, penalized slack)")
                # Each retry asked for by the critic tries the next relaxation the model found.
                pick = min(attempt, len(found) - 1) if found else None
                if pick is not None:
                    best = found[pick]
                    moved = limits.model_copy(update={best.limit: best.relaxed if best.limit != "max_hires_per_month" else int(best.relaxed)})
                    months = planning.evaluate_full(planner, planning.country_params(plan.get("scenario"), country, moved.model_copy(update={"budget_usd": None})), "optimized")[3]
                    options.append(
                        planning.option(country, "relaxed", best.kpis, best.peak_hires, [best.note] if best.note else [], limits, automation, best.as_dict(), months)
                    )
            if attempt >= 1 and limits.max_hires_per_month is not None:
                lifted = planning.country_params(plan.get("scenario"), country, limits.model_copy(update={"max_hires_per_month": None}))
                kpis, peak, notes, months = planning.evaluate_full(planner, lifted, "optimized")
                options.append(planning.option(country, "lifted_cap", kpis, peak, notes, limits, automation, months=months))
    for record in options:
        facts.update({f"{k}.{record['name']}.{name}": value for name, value in planning.option_facts(record).items()})

    result = {
        "country": country,
        "version": version,
        "attempt": attempt,
        "options": options,
        "strict_issues": strict_issues,
        "relaxations": relaxations,
        # The plan the year started with, so every option can show what it changes.
        "baseline": base_kpis,
        "baseline_schedule": planning.schedule(base_outcome.plan),
    }
    detail = f"{country}: {len(options)} options"
    if strict_issues:
        detail += f"; your limits cannot all be met ({'; '.join(strict_issues)})"
    if relaxations:
        detail += f"; smallest change: {relaxations[0]['description']}"
    if feedback:
        detail += f"; retry {attempt} after: {'; '.join(feedback)}"
    trace = [event("Optimizer", "ran the options", detail, seconds=time.perf_counter() - started, tool=", ".join(tools), country=country, model="no model (optimizer and simulator)")]
    return {"country_results": {country: result}, "fact_ledger": facts, "trace": trace}


# Decision critic


def rule_issues(services: CrewServices, record: dict[str, Any], limits: Constraints) -> list[str]:
    rules = services.settings.rules
    kpis = record["kpis"]
    issues = []
    if kpis["max_wait_days"] > rules.max_wait_days:
        issues.append(f"an advertiser waits {kpis['max_wait_days']:.0f} days, above the {rules.max_wait_days} day limit")
    if kpis["peak_load"] > rules.capacity_per_agent or kpis["max_overflow"] > 0:
        issues.append(f"an agent holds more than {rules.capacity_per_agent} advertisers")
    if record["name"] in {"within_limits", "relaxed"}:
        check = limits
        if record.get("relaxation"):
            check = limits.model_copy(update={record["relaxation"]["limit"]: record["relaxation"]["relaxed"]})
        issues += meets(kpis, check, record.get("peak_hires"))
    return issues


def preferred(result: dict[str, Any], limits: Constraints) -> str:
    names = [o["name"] for o in result["options"]]
    if "update" in names:
        return "update"
    if limits.count():
        if not result["strict_issues"]:
            return "within_limits"
        if "relaxed" in names:
            return "relaxed"
        return "within_limits"
    return "recommended"


def option_cards(results: dict[str, dict[str, Any]], countries: list[str]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Options summed over the countries, one card per option name, with "all." facts."""
    names: list[str] = []
    for country in countries:
        for record in results[country]["options"]:
            if record["name"] not in names:
                names.append(record["name"])
    fallback = {"relaxed": "within_limits", "lifted_cap": "within_limits", "within_limits": "recommended"}
    cards, facts = [], {}
    for name in names:
        rows, labels = [], set()
        for country in countries:
            by_name = {o["name"]: o for o in results[country]["options"]}
            pick = name
            while pick not in by_name and pick in fallback:
                pick = fallback[pick]
            if pick in by_name:
                rows.append(by_name[pick])
                labels.add(by_name[pick]["label"])
        label = next(o["label"] for c in countries for o in results[c]["options"] if o["name"] == name)
        total = {m: sum(r["kpis"][m] for r in rows) for m in ("net_value", "total_cost", "uplift_value", "served", "arrivals", "dropped", "hires", "notices", "start_agents", "end_agents")}
        total["service_rate"] = total["served"] / total["arrivals"] if total["arrivals"] else 1.0
        total["max_wait_days"] = max(r["kpis"]["max_wait_days"] for r in rows)
        served = sum(r["kpis"]["served"] for r in rows)
        total["mean_wait_days"] = sum(r["kpis"]["mean_wait_days"] * r["kpis"]["served"] for r in rows) / served if served else 0.0
        card = {"name": name, "label": label, "totals": total, "relaxed": any(r["relaxed"] for r in rows), "countries": [r["country"] for r in rows]}
        card["relaxations"] = [f"{r['country']}: {r['relaxation']['description']}" for r in rows if r.get("relaxation")]
        autos = [r["automation"] for r in rows if r.get("automation")]
        if autos:
            card["automation"] = {m: sum(a[m] for a in autos) for m in ("handled", "value_low", "value_high", "net_value_low", "net_value_high")}
        cards.append(card)
        facts.update({f"all.{name}.{m}": v for m, v in total.items()})
    return merge_identical(cards), facts


# When two options lead to the same plan, the one that answers the request is kept. A relaxed
# option exists only when the limits cannot all be met, so it outranks "within your limits".
KEEP_ORDER = ("relaxed", "within_limits", "lifted_cap", "update", "recommended", "profit_only", "keep")


def merge_identical(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One card per distinct plan; a dropped twin is named on the card that stays."""
    def same(a: dict[str, Any], b: dict[str, Any]) -> bool:
        keys = ("net_value", "total_cost", "served", "hires", "notices")
        return all(abs(a["totals"][k] - b["totals"][k]) < 0.5 for k in keys) and a.get("automation") == b.get("automation")

    ranked = sorted(cards, key=lambda c: KEEP_ORDER.index(c["name"]) if c["name"] in KEEP_ORDER else len(KEEP_ORDER))
    kept: list[dict[str, Any]] = []
    for card in ranked:
        twin = next((k for k in kept if same(k, card)), None)
        if twin is None:
            kept.append({**card, "same_as": []})
        else:
            twin["same_as"].append(card["label"])
    order = [c["name"] for c in cards]
    return sorted(kept, key=lambda c: order.index(c["name"]))


def critic_node(services: CrewServices, state: CrewState) -> dict[str, Any]:
    started = time.perf_counter()
    plan = state["plan"]
    countries = plan["countries"]
    limits = Constraints.model_validate(state.get("user_constraints") or {})
    results = state["country_results"]
    retries = dict(state.get("retry_counts") or {"optimizer": 0, "writer": 0})
    max_retries = services.crew.graph.optimizer_retries

    problems: dict[str, list[str]] = {}
    for country in countries:
        result = results[country]
        chosen = next(o for o in result["options"] if o["name"] == preferred(result, limits))
        issues = rule_issues(services, chosen, limits)
        if issues:
            problems[country] = issues
    choice = services.router().choose("decide", difficulty_of(state), state.get("mode", "best_quality"))
    routing = [{"agent": "Decision critic", **choice.as_dict()}]

    if problems and retries.get("optimizer", 0) < max_retries:
        retries["optimizer"] = retries.get("optimizer", 0) + 1
        detail = "; ".join(f"{c}: {', '.join(i)}" for c, i in problems.items())
        trace = [event("Decision critic", f"sent {len(problems)} {'country' if len(problems) == 1 else 'countries'} back to the Optimizer (retry {retries['optimizer']} of {max_retries})", detail, choice, time.perf_counter() - started)]
        return {"retry_counts": retries, "critic_feedback": problems, "trace": trace, "routing_trace": routing}

    cards, facts = option_cards(results, countries)
    for card in cards:
        rows = [o for c in countries for o in results[c]["options"] if o["name"] == card["name"]]
        card["issues"] = sorted({i for r in rows for i in rule_issues(services, r, limits)})
    names = [c["name"] for c in cards]
    wanted = "update" if "update" in names else ("relaxed" if "relaxed" in names else ("within_limits" if limits.count() else "recommended"))
    if limits.count() and "within_limits" in names and not next(c for c in cards if c["name"] == "within_limits")["issues"]:
        wanted = "within_limits"
    clean = [c for c in cards if not c["issues"]]
    chosen = wanted if wanted in names else names[0]
    if next(c for c in cards if c["name"] == chosen)["issues"] and clean and not limits.count():
        chosen = max(clean, key=lambda c: c["totals"]["net_value"])["name"]
    open_issues = [f"{c}: {i}" for c, issues in problems.items() for i in issues]
    detail = f"chose {next(c['label'] for c in cards if c['name'] == chosen)}; {len(cards)} options compared"
    if open_issues:
        detail += f"; {len(open_issues)} open issues passed on after {retries.get('optimizer', 0)} retries"
    trace = [event("Decision critic", "compared the options", detail, choice, time.perf_counter() - started, tool="rule checks")]
    return {"options": cards, "chosen": chosen, "open_issues": open_issues, "critic_feedback": {}, "fact_ledger": facts, "retry_counts": retries, "trace": trace, "routing_trace": routing}


# Writer


def _money_line(prefix: str) -> str:
    return (
        f"net value {{{{{prefix}.net_value}}}} for a staffing cost of {{{{{prefix}.total_cost}}}}, "
        f"{{{{{prefix}.service_rate}}}} of eligible advertisers served, "
        f"{{{{{prefix}.hires}}}} hires and {{{{{prefix}.notices}}}} notices"
    )


def template_draft(state: CrewState) -> str:
    """The answer with placeholders only, written without a model. Every number comes from the ledger."""
    plan = state["plan"]
    countries = plan["countries"]
    cards = state["options"]
    chosen = next(c for c in cards if c["name"] == state["chosen"])
    audience = state.get("audience", "manager")
    results = state["country_results"]
    analyses = state.get("analyses") or {}
    many = len(countries) > 1
    lines: list[str] = []

    def option_prefix(country: str, name: str) -> str:
        return f"{key(country)}.{name}" if not many else f"all.{name}"

    if audience == "country_manager":
        for country in countries:
            k = key(country)
            name = chosen["name"] if any(o["name"] == chosen["name"] for o in results[country]["options"]) else "recommended"
            p = f"{k}.{name}"
            lines += [
                f"## Staffing memo: {country}",
                "",
                f"**Plan:** {chosen['label'].lower()}, from {{{{{p}.start_agents}}}} to {{{{{p}.end_agents}}}} agents over the year.",
                "",
                "**Hires and notices**",
                f"* {{{{{p}.hires}}}} hires and {{{{{p}.notices}}}} notices in the year; new agents start one month after the decision.",
                "",
                "**Cost and value**",
                f"* Staffing cost {{{{{p}.total_cost}}}}, uplift value {{{{{p}.uplift_value}}}}, net value {{{{{p}.net_value}}}}.",
                "",
                "**Why**",
                *[f"* {f}" for f in analyses.get(country, {}).get("findings", [])],
                "",
                "**Service**",
                f"* {{{{{p}.service_rate}}}} of eligible advertisers served; the average wait is {{{{{p}.mean_wait_days}}}} days.",
                "",
            ]
    elif audience == "people":
        lines += [f"## Hiring and notice plan: {', '.join(countries)}", ""]
        for country in countries:
            k = key(country)
            record = next((o for o in results[country]["options"] if o["name"] == chosen["name"]), results[country]["options"][0])
            p = f"{k}.{record['name']}"
            lines += [
                (f"**{country}, {record['label'].lower()}:** {{{{{p}.hires}}}} hires and {{{{{p}.notices}}}} notices in the year, "
                f"from {{{{{p}.start_agents}}}} to {{{{{p}.end_agents}}}} agents."),
                "",
            ]
            months = record.get("schedule") or {}
            steps = []
            for index, month in enumerate(MONTHS, start=1):
                hires = (months.get("hires") or [0] * 12)[index - 1]
                notices = (months.get("notices") or [0] * 12)[index - 1]
                if hires:
                    steps.append(f"* {month}: decide {{{{{p}.hires_m{index}}}}} hires; they start the month after.")
                if notices:
                    steps.append(f"* {month}: give {{{{{p}.notices_m{index}}}}} notices; those agents leave the month after.")
            lines += steps or ["* No hires or notices are needed this year."]
            lines.append("")
        lines += [
            ("* Lead times: a new agent starts {{rules.ramp_months}} month after the hiring decision, and an agent under notice "
            "leaves {{rules.notice_months}} month after the notice."),
            "* Check local notice rules and plan the communication before the first notice is given.",
        ]
    elif audience == "finance":
        p = option_prefix(countries[0], chosen["name"])
        base = "all.recommended" if many else f"{key(countries[0])}.base"
        lines += [
            f"## Finance note: {', '.join(countries)}",
            "",
            f"* **{chosen['label']}:** staffing cost {{{{{p}.total_cost}}}}, uplift value {{{{{p}.uplift_value}}}}, net value {{{{{p}.net_value}}}}.",
            f"* The recommended plan before this request: net value {{{{{base}.net_value}}}} for a cost of {{{{{base}.total_cost}}}}.",
        ]
        for card in cards:
            if card["name"] != chosen["name"]:
                q = option_prefix(countries[0], card["name"])
                lines.append(f"* {card['label']}: net value {{{{{q}.net_value}}}} for a cost of {{{{{q}.total_cost}}}}.")
    else:
        p = option_prefix(countries[0], chosen["name"])
        lines += [f"**Recommendation: {chosen['label'].lower()}** for {', '.join(countries)}.", "", f"This plan gives {_money_line(p)}.", ""]
        if many:
            for country in countries:
                name = chosen["name"] if any(o["name"] == chosen["name"] for o in results[country]["options"]) else "recommended"
                lines.append(f"* {country}: {_money_line(f'{key(country)}.{name}')}.")
            lines.append("")
        findings = [f for country in countries for f in analyses.get(country, {}).get("findings", [])]
        if findings:
            lines += ["What the data shows:", *[f"* {f}" for f in findings]]

    for country in countries:
        k = key(country)
        result = results[country]
        if result["strict_issues"]:
            lines += ["", f"Your limits cannot all be met in {country}."]
            if result["relaxations"]:
                relax = result["relaxations"][0]
                name = {"max_hires_per_month": "hiring_cap", "budget_usd": "budget", "service_goal": "service_goal_share"}[relax["limit"]]
                verb = {"max_hires_per_month": "raise the hiring cap", "budget_usd": "raise the budget", "service_goal": "lower the service goal"}[relax["limit"]]
                if any(o["name"] == "relaxed" for o in result["options"]):
                    lines.append(
                        f"* The smallest change is to {verb} from {{{{{k}.relaxed.original_{name}}}}} to {{{{{k}.relaxed.relaxed_{name}}}}}; "
                        f"this option is marked as relaxed."
                    )
        auto = next((o.get("automation") for o in result["options"] if o["name"] == chosen["name"] and o.get("automation")), None)
        if auto:
            p = f"{k}.{chosen['name']}"
            lines += [
                "",
                (f"With {{{{{p}.ai_share}}}} of advertisers handled by an AI assistant in {country}, the human team serves the rest. "
                f"The AI handles about {{{{{p}.ai_handled}}}} advertisers; with no data on the uplift it earns, net value is between "
                f"{{{{{p}.net_value_ai_low}}}} and {{{{{p}.net_value_ai_high}}}}."),
            ]
        if plan.get("intent") == "monthly_update":
            month = MONTHS[int(state["monthly"]["as_of"]) - 1]
            lines += ["", f"For {month} in {country}: hire {{{{{k}.next.hires}}}} (planned {{{{{k}.next.planned_hires}}}}) and give {{{{{k}.next.notices}}}} notices (planned {{{{{k}.next.planned_notices}}}})."]
    if state.get("open_issues"):
        lines += ["", "Still open after two more tries:", *[f"* {i}" for i in state["open_issues"]]]
    lines += ["", "Nothing is applied until you approve an option."]
    return "\n".join(lines).strip()


def writer_node(services: CrewServices, state: CrewState) -> dict[str, Any]:
    started = time.perf_counter()
    retries = dict(state.get("retry_counts") or {})
    attempt = retries.get("writer", 0)
    countries = state["plan"]["countries"]
    difficulty = difficulty_of(state)
    mode = state.get("mode", "best_quality")
    router = services.router()
    previous = state.get("writer_model")
    if attempt >= 2 and previous:
        choice = router.escalate(RouteChoice(**previous), difficulty, mode)
    elif previous and attempt == 1:
        choice = RouteChoice(**{**previous, "reason": "retry with the same model after a failed number check"})
    else:
        choice = router.choose("write", difficulty, mode)
    ledger = FactLedger(dict(state.get("fact_ledger") or {}))
    template = template_draft(state)
    calls = []
    draft = template
    if choice.model != OFFLINE:
        audience = AUDIENCE_LABELS[state.get("audience", "manager")]
        notes = [
            f"Write for the {audience.lower()}.",
            f"Recommend: {next(c['label'] for c in state['options'] if c['name'] == state['chosen'])}.",
            "Write every number as a {{placeholder}} from the facts. Say that nothing is applied until the manager approves.",
            *state.get("open_issues", []),
        ]
        feedback = ""
        if state.get("unverified"):
            feedback = "These numbers had no source; use placeholders instead: " + ", ".join(state["unverified"])
        output = ToolOutput("crew_options", dict(ledger.facts), notes=notes)
        reading = RoutedQuery(intent=Intent.what_if if state["plan"].get("scenario") else Intent.plan_lookup, countries=countries)
        text, record = services.registry.client(choice.model).call(
            "compose", state["request"], reading, [output], ledger.formatted(), feedback, agent="Writer", step="write"
        )
        calls.append(record.as_dict())
        if record.ok and text:
            draft = text
    rendered, missing = ledger.render(draft)
    trace = [event("Writer", f"wrote the answer (attempt {attempt + 1})", f"for the {AUDIENCE_LABELS[state.get('audience', 'manager')].lower()}", choice, time.perf_counter() - started)]
    return {
        "draft": draft,
        "answer": rendered,
        "unverified": missing,
        "writer_model": choice.as_dict(),
        "trace": trace,
        "routing_trace": [{"agent": "Writer", "attempt": attempt + 1, **choice.as_dict()}],
        "calls": calls,
    }


def number_check_node(services: CrewServices, state: CrewState) -> dict[str, Any]:
    """The Decision critic's second job: every number in the answer must come from the ledger."""
    started = time.perf_counter()
    ledger = FactLedger(dict(state.get("fact_ledger") or {}))
    limits = Constraints.model_validate(state.get("user_constraints") or {})
    # Option labels ("at least 98% served") and the manager's own limits are sources too.
    labels = [c["label"] for c in state.get("options", [])]
    context = " ".join([state.get("request", ""), *limits.describe(), *state.get("open_issues", []), *labels])
    countries = state["plan"]["countries"]
    answer = state.get("answer", "")
    unsupported = list(state.get("unverified") or []) + ledger.verify(answer, context=context, countries=countries)
    retries = dict(state.get("retry_counts") or {})
    limit = services.crew.graph.writer_retries
    if unsupported and retries.get("writer", 0) < limit:
        retries["writer"] = retries.get("writer", 0) + 1
        next_try = "a stronger model" if retries["writer"] >= 2 else "the same model"
        trace = [event("Decision critic", f"sent the draft back to the Writer (retry {retries['writer']} of {limit}, {next_try})", f"{len(unsupported)} numbers without a source", seconds=time.perf_counter() - started, tool="FactLedger", model="no model (FactLedger)")]
        return {"retry_counts": retries, "unverified": unsupported, "trace": trace}
    flagged: list[str] = []
    if unsupported:
        kept = []
        for line in answer.splitlines():
            sentences = re.split(r"(?<=[.!?])\s+", line)
            good = [s for s in sentences if not any(token in s for token in unsupported)]
            flagged += [s for s in sentences if s not in good]
            if good or not line.strip():
                kept.append(" ".join(good))
        answer = "\n".join(kept).strip()
        answer += f"\n\n_{len(flagged)} {'sentence was' if len(flagged) == 1 else 'sentences were'} left out because a figure in it could not be traced to the planner._"
    detail = "every number traced to the planner" if not unsupported else f"{len(flagged)} sentences removed after {retries.get('writer', 0)} retries"
    trace = [event("Decision critic", "checked every number", detail, seconds=time.perf_counter() - started, tool="FactLedger", model="no model (FactLedger)")]
    return {"answer": answer, "unverified": unsupported, "flagged": flagged, "retry_counts": retries, "trace": trace, "approval_status": "Waiting for approval"}


def params_for(scenario: dict[str, Any] | None) -> ScenarioParams:
    return ScenarioParams.model_validate(scenario or {})
