"""Run a small live mimOE regression suite and record the actual tool traces."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import HumanMessage
from langsmith import tracing_context
from openai import APIConnectionError, APIStatusError

from .agent import Agent, make_model, model_error
from .cli import EXAMPLES, render
from .data import Recording, Second, load_recording

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
    parser.add_argument("--extended", action="store_true",
                        help="Also repeat the paired investigation and test paraphrases and event choice")
    parser.add_argument("--output", type=Path, help="Save actual results as JSON")
    parser.add_argument("--capture-responses", action="store_true",
                        help="Include raw model replies and completion metadata in the JSON report")
    args = parser.parse_args()
    if args.capture_responses and not args.output:
        parser.error("--capture-responses requires --output")
    try:
        model = make_model()
        model_id = model.model_name
        loaded_ids = [entry.id for entry in model.root_client.models.list().data]
        if loaded_ids != [model_id]:
            print(f"Expected exactly one loaded model ({model_id}). mimOE lists: {loaded_ids}.")
            return 1
        if args.smoke:
            with tracing_context(enabled=False):
                answer = model.bind(extra_body={"max_tokens": 20}).invoke([HumanMessage(content="Say hello in one sentence.")])
            served = answer.response_metadata.get("model_name")
            if served != model_id:
                print(f"Model mismatch: requested {model_id}, served {served}.")
                return 1
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

    def execute(name, agent, question, expected, status, expected_args, goal=None):
        responses = []
        served_models = set()

        def observe(answer):
            served_models.add(answer.response_metadata.get("model_name"))
            if args.capture_responses:
                responses.append({"content": answer.content, "metadata": answer.response_metadata})

        run = agent.run(question, on_response=observe)
        actual = [step.action for step in run.steps]
        shown = [step.action for step in run.shown]
        expected_goal = goal or ("compare" if status == "clarify" else
                                 "unsupported" if status == "unsupported" else expected[-1])
        passed = (served_models == {model_id} and run.goal == expected_goal
                  and run.status == status and actual == expected and shown == expected
                  and (expected_args is None or run.steps[-1].args == expected_args))
        print(f"{'PASS' if passed else 'FAIL'} {name}: goal={run.goal}, {actual} "
              f"({run.status}, {run.stop_reason}, {run.elapsed_s:.2f}s, {len(run.errors)} rejected replies)",
              flush=True)
        outcomes.append({"name": name, "recording": agent.recording.key, "question": question,
                         "expected_tools": expected, "expected_args": expected_args, "expected_status": status,
                         "served_models": sorted(str(name) for name in served_models),
                         "expected_goal": expected_goal,
                         "passed": passed, "run": asdict(run), "displayed": render(run)})
        if args.capture_responses:
            outcomes[-1]["responses"] = responses
        return run

    for name, key, question, expected, status, expected_args in CASES:
        execute(name, Agent(model, load_recording(key)), question, expected, status, expected_args,
                goal="motion" if name == "no_rise_to_check" else None)
    agent = Agent(model, load_recording("rising"))
    execute("followup_setup", agent, "Find sustained heart-rate elevations.", ["heart"], "ok", None)
    execute("followup", agent, "How much wrist movement was recorded around that first event?",
            ["motion"], "ok", {"event": 1})

    passed = sum(item["passed"] for item in outcomes)
    print(f"{passed}/{len(outcomes)} live cases passed ({model_id}).")
    original = outcomes.copy()
    print(f"Correct goal selections: {sum(c['run']['goal'] == c['expected_goal'] for c in original)}/"
          f"{len(original)}. Completed tools: {sum(len(c['run']['steps']) for c in original)}.")
    if args.extended:
        for attempt in (2, 3):
            for key, expected, event_args in (
                ("rising", ["heart", "motion"], {"event": 1}), ("steady", ["heart"], None),
            ):
                execute(f"{key}_pair_{attempt}", Agent(model, load_recording(key)), EXAMPLES[2],
                        expected, "ok", event_args, goal="motion")
        for name, key, question, tools, args_ in (
            ("coverage_paraphrase", "offset", "Did either channel miss any samples?", ["quality"], None),
            ("motion_paraphrase", "rising", "Was the biggest pulse spike accompanied by wrist activity?",
             ["heart", "motion"], {"event": 1}),
            ("comparison_paraphrase", "offset",
             "Contrast 00:00–01:00 with 01:00–02:00 for heart rate and wrist movement.",
             ["compare"], {"a": [0, 60], "b": [60, 120]}),
        ):
            execute(name, Agent(model, load_recording(key)), question, tools, "ok", args_)
    extended = outcomes[len(original):]
    if args.extended:
        # Swap the peak heights so the same chronological request must
        # choose a different observed event ID.
        for first, second in ((80, 90), (90, 80)):
            heart = [60] * 120 + [first] * 30 + [60] * 20 + [second] * 30
            recording = Recording(f"synthetic_{first}_{second}", "Synthetic event choice", "",
                                  tuple(Second(bpm, 0.03, 1, 32) for bpm in heart), 1, 32)
            earlier_id, later_id = (2, 1) if first < second else (1, 2)
            for reference, event_id in (("earlier", earlier_id), ("later", later_id)):
                execute(f"observed_{reference}_{first}_{second}", Agent(model, recording),
                        f"Check wrist movement at the {reference} heart-rate event.",
                        ["heart", "motion"], "ok", {"event": event_id})
        execute("ambiguous_observed_event", Agent(model, recording),
                "Check wrist movement at an event.", ["heart"], "clarify", None, goal="motion")
    event_choices = outcomes[len(original) + len(extended):]
    if extended:
        print(f"{sum(c['passed'] for c in extended)}/{len(extended)} extended cases passed.")
    if event_choices:
        print(f"{sum(c['passed'] for c in event_choices)}/{len(event_choices)} synthetic event-choice cases passed.")
    if args.output:
        report = {"model": model_id, "cases": original}
        if extended:
            report["extended_cases"] = extended
        if event_choices:
            report["synthetic_event_cases"] = event_choices
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(not all(c["passed"] for c in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
