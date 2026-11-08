"""Run a small live mimOE regression suite and record the actual tool traces."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import HumanMessage
from langsmith import tracing_context
from openai import APIConnectionError, APIStatusError

from .agent import Agent, DEFAULT_MODEL, make_model, model_error
from .cli import EXAMPLES, render
from .data import load_recording

# Each case includes an expected trace. Paired recordings reuse the same question.
CASES = [
    ("coverage", "offset", EXAMPLES[0], ["quality"], "ok", None),
    ("elevations", "rising", EXAMPLES[1], ["heart"], "ok", None),
    ("motion_at_rise", "rising", EXAMPLES[2], ["heart", "motion"], "ok", {"event": 1}),
    ("no_rise_to_check", "steady", EXAMPLES[2], ["heart"], "ok", None),
    ("event_vs_baseline", "rising", EXAMPLES[3], ["heart", "compare"], "ok", {"event": 1}),
    ("explicit_windows", "offset", "Contrast minute 0 to 1 with minute 1 to 2 for pulse and motion.",
     ["compare"], "ok", {"a": [0, 60], "b": [60, 120]}),
    ("quality_paraphrase", "offset", "Do both streams cover the full recording?", ["quality"], "ok", None),
    ("unsupported", "rising", "Can you diagnose a heart condition from this?", [], "unsupported", None),
    ("ambiguous", "rising", "Compare two time windows for me.", [], "clarify", None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Only check live Chat Completions via LangChain")
    parser.add_argument("--output", type=Path, help="Save actual results as JSON")
    args = parser.parse_args()
    try:
        model = make_model()
        if args.smoke:
            with tracing_context(enabled=False):
                answer = model.bind(max_tokens=20).invoke([HumanMessage(content="Say hello in one sentence.")])
            if not isinstance(answer.content, str) or not answer.content.strip():
                print("mimOE returned an empty or non-text completion.")
                return 1
            print(f"Live LangChain/mimOE response ({model.model_name}): {answer.content}")
            return 0
    except ValueError as error:
        print(error)
        return 1
    except (APIConnectionError, APIStatusError) as error:
        print(model_error(error))
        return 1

    outcomes = []

    def execute(name, agent, question, expected, status, expected_args):
        run = agent.run(question)
        actual = [step.action for step in run.steps]
        shown = [step.action for step in run.shown]
        passed = (run.status == status and actual == expected and shown == expected
                  and (expected_args is None or run.steps[-1].args == expected_args))
        print(f"{'PASS' if passed else 'FAIL'} {name}: {actual} ({run.status}, {run.elapsed_s:.2f}s, "
              f"{len(run.errors)} invalid actions)", flush=True)
        outcomes.append({"name": name, "recording": agent.recording.key, "question": question,
                         "expected_tools": expected, "expected_args": expected_args, "expected_status": status,
                         "passed": passed, "run": asdict(run), "displayed": render(run)})
        return run

    for name, key, question, expected, status, expected_args in CASES:
        execute(name, Agent(model, load_recording(key)), question, expected, status, expected_args)
    agent = Agent(model, load_recording("rising"))
    execute("followup_setup", agent, "Find sustained heart-rate elevations.", ["heart"], "ok", None)
    execute("followup", agent, "How much wrist movement was recorded around that first event?",
            ["motion"], "ok", {"event": 1})

    model_id = os.getenv("MIMOE_MODEL", DEFAULT_MODEL)
    passed = sum(item["passed"] for item in outcomes)
    print(f"{passed}/{len(outcomes)} live cases passed ({model_id}).")
    if args.output:
        args.output.write_text(json.dumps({"model": model_id, "cases": outcomes}, indent=2) + "\n",
                               encoding="utf-8")
    return int(passed != len(outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
