"""Command line: run the app, ask one question, or run the evaluation.

    staffpilot-crew app [--sample] [--port 7861] [--offline]
    staffpilot-crew ask "Plan for Brazil with at most 5 hires a month" [--sample] [--mode lowest_cost]
    staffpilot-crew eval [--sample] [--online] [--yes]
"""

from __future__ import annotations

import argparse
import os
import sys

from staffpilot.config import load_env_file, load_settings
from staffpilot.data import DatasetError

from staffpilot_crew.config import load_crew_settings
from staffpilot_crew.graph import CrewRunner, sqlite_checkpointer
from staffpilot_crew.router import MODES
from staffpilot_crew.services import CrewServices


def make_services(sample: bool) -> tuple[CrewServices, str, str]:
    from staffpilot.planner import Planner

    from staffpilot_crew.sample import sample_planner

    crew = load_crew_settings()
    if sample:
        return CrewServices.build(sample_planner(), crew), "Data: a synthetic sample with three made up countries.", "sample"
    settings = load_settings()
    if not settings.data_path().exists():
        print(f"The course file was not found at {settings.data_path()}; using the synthetic sample instead.", file=sys.stderr)
        return CrewServices.build(sample_planner(), crew), "Data: a synthetic sample (the course file was not found).", "sample"
    try:
        planner = Planner(settings)
    except DatasetError as error:
        sys.exit(f"The course file could not be read: {error}")
    return CrewServices.build(planner, crew), f"Data: the course dataset, planning {planner.year}.", "course"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="staffpilot-crew", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    app = sub.add_parser("app", help="serve the web app")
    app.add_argument("--port", type=int, default=7861)
    ask = sub.add_parser("ask", help="run one request until it waits for approval")
    ask.add_argument("request")
    ask.add_argument("--mode", choices=MODES, default="best_quality")
    ev = sub.add_parser("eval", help="compare the router with each single model")
    ev.add_argument("--online", action="store_true", help="also run the models that have keys (costs money)")
    ev.add_argument("--yes", action="store_true", help="do not ask before an online run")
    for command in (app, ask, ev):
        command.add_argument("--sample", action="store_true", help="use the synthetic sample instead of the course file")
        command.add_argument("--offline", action="store_true", help="never call a model, even when keys exist")
    args = parser.parse_args(argv)

    load_env_file()
    if args.offline:
        os.environ["STAFFPILOT_OFFLINE"] = "1"
    services, note, question_set = make_services(args.sample)

    if args.command == "app":
        from staffpilot_crew.app import EXAMPLES, serve

        runner = CrewRunner(services, sqlite_checkpointer(services.crew.paths.checkpoints))
        serve(services, runner, note, EXAMPLES[question_set], args.port)
    elif args.command == "ask":
        values = CrewRunner(services).run(args.request, mode=args.mode)
        for event in values["trace"]:
            model = f" [{event['model']}]" if event.get("model") else ""
            print(f"* {event['agent']}{model}: {event['action']}. {event['detail']}")
        print()
        print(values.get("answer", ""))
        print()
        print(f"Status: {values.get('approval_status')}")
    else:
        from staffpilot_crew import evaluation

        questions = evaluation.load_questions(question_set)
        if args.online:
            setups = evaluation.setups(services, online=True)
            print(f"{len(setups)} setups x {len(questions)} requests = {len(setups) * len(questions)} graph runs, each with several model calls.")
            if not args.yes and input("This spends money on the providers with keys. Continue? [y/N] ").strip().lower() != "y":
                sys.exit("Stopped; nothing was spent.")
        rows, summary = evaluation.evaluate(services, question_set, online=args.online, progress=print)
        folder = evaluation.save(rows, summary, services)
        print(summary.to_string(index=False))
        print(f"Saved in {folder}")


if __name__ == "__main__":
    main()
