"""Guided CLI and human-readable rendering of validated tool evidence."""
import argparse
import json
from dataclasses import asdict

from .agent import Agent, make_model
from .data import catalog, load_recording

EXAMPLES = [
    "Check this wearable recording for missing measurements or other data-quality problems.",
    "Find the most significant periods of elevated heart rate and tell me when they occurred.",
    "Investigate the largest heart-rate increase. Was it accompanied by increased wrist movement?",
    "Compare the largest heart-rate elevation with the first two minutes. How did heart rate and movement differ?",
]


def clock(second):
    return f"{second // 60:02d}:{second % 60:02d}"


def interval(window):
    return f"{clock(window[0])} to {clock(window[1])}"


def metric(stats, unit, decimals):
    count = f"{stats['valid_s']}/{stats['total_s']} valid seconds"
    if stats["status"] != "ok":
        return f"insufficient coverage ({count})"
    return (f"mean {stats['mean']:.{decimals}f}, median {stats['median']:.{decimals}f}, "
            f"SD {stats['sd']:.{decimals}f} {unit} ({count})")


def render_window(name, data):
    if data is None:
        return f"  {name}: outside excerpt."
    return (f"  {name} ({interval(data['window_s'])})\n"
            f"    Heart rate: {metric(data['heart_rate_bpm'], 'bpm', 1)}\n"
            f"    Wrist movement: {metric(data['wrist_motion_g'], 'g RMS', 4)}")


def render(run):
    lines = [run.message] if run.message else []
    for step in run.shown:
        data = step.result
        if step.action == "quality":
            lines.append("Signal quality")
            for name, sensor in data["signals"].items():
                label = name.replace("_", " ").capitalize()
                if not sensor["available"]:
                    lines.append(f"  {label}: unavailable.")
                    continue
                lines.append(f"  {label}: {sensor['sample_rate_hz']} Hz, {sensor['coverage']:.1%} "
                             f"usable coverage, {sensor['missing_s']} unusable seconds across "
                             f"{sensor['gap_count']} gaps, {sensor['partial_bins']} partial sample bins.")
                for gap in sensor["largest_gaps_s"]:
                    lines.append(f"    Gap: {interval(gap)}.")
                if sensor["unexpected_rate"]:
                    lines.append(f"    Expected {sensor['expected_e4_rate_hz']} Hz for this E4 signal.")
            if data["issue_count"]:
                lines.append(f"  Input issues: {data['issue_count']}. Examples: {data['examples']}.")
            if any(item["coverage"] < 1 for item in data["signals"].values()):
                lines.append("  Tools exclude unusable seconds and withhold statistics with low coverage.")
        elif step.action == "heart":
            overall = data["overall"]
            lines.append(f"Heart rate, {interval(overall['window_s'])}: {metric(overall, 'bpm', 1)}.")
            if overall["status"] == "ok":
                lines.append(f"  Range {overall['min']:.1f} to {overall['max']:.1f} bpm.")
            trend = data["last_vs_first_minute_bpm"]
            if trend is not None:
                lines.append(f"  Last vs. first minute mean: {trend:+.1f} bpm.")
            if data["status"] != "ok":
                lines.append(f"  Event detection unavailable. {data['reason']}")
                continue
            lines.append(f"  Baseline median {data['baseline']['median']:.1f} bpm "
                         f"({data['baseline']['valid_s']}/{data['baseline']['total_s']} valid seconds). "
                         f"Event rule: at least {data['minimum_duration_s']} s at or above "
                         f"{data['threshold_bpm']:.1f} bpm.")
            if not data["events"]:
                lines.append("  No sustained elevation met this recording-relative rule.")
            elif data["event_count"] > len(data["events"]):
                lines.append(f"  Showing {len(data['events'])} of {data['event_count']} events by peak.")
            for event in data["events"]:
                lines.append(f"  Event {event['id']}: {interval(event['window_s'])}, "
                             f"mean {event['mean_bpm']:.1f}, peak {event['peak_bpm']:.1f} bpm "
                             f"({event['peak_above_baseline_bpm']:+.1f} bpm vs. baseline).")
        elif step.action == "motion":
            lines.append("Wrist movement around the selected interval")
            lines.extend(render_window(label, data[key])
                         for label, key in (("Before", "before"), ("During", "during"), ("After", "after")))
            if data["relative_motion"] is None:
                lines.append("  Insufficient comparable wrist movement before and during.")
            else:
                lines.append(f"  Movement during the window was {data['relative_motion']} than before "
                             f"({data['during_minus_before_motion_g']:+.4f} g RMS)."
                             if data["relative_motion"] != "similar" else
                             f"  Movement during the window was similar to before "
                             f"({data['during_minus_before_motion_g']:+.4f} g RMS).")
            lines.append(f"  {data['note']}")
        else:
            lines.append("Interval comparison (B minus A)")
            lines.extend(render_window(label, data[key]) for label, key in (("A", "a"), ("B", "b")))
            for field, label, unit, digits in (
                ("b_minus_a_bpm", "Mean heart rate", "bpm", 1),
                ("b_minus_a_motion_g", "Mean wrist movement", "g RMS", 4),
            ):
                value = data[field]
                lines.append(f"  {label}: {value:+.{digits}f} {unit}." if value is not None else
                             f"  {label}: insufficient coverage for a difference.")
    return "\n".join(lines)


def show_step(tool, step):
    print(f"  → {tool} {json.dumps(step.args, separators=(',', ':'))}", flush=True)


def menu():
    entries = catalog()
    keys = list(entries)
    for i, (key, entry) in enumerate(entries.items(), 1):
        print(f"  {i}. {entry['title']} [{key}] — {entry['description']}")
    while True:
        choice = input("Choose a recording (q to quit): ").strip()
        if choice.lower() == "q":
            return None
        if choice in keys:
            return choice
        if choice in [str(i) for i in range(1, len(keys) + 1)]:
            return keys[int(choice) - 1]
        print("Enter a listed recording number or key.")


def interactive(model, key):
    print("WEARABLE-DATA INVESTIGATION AGENT\nPowered by mimOE + LangChain\n")
    print("Investigate recorded heart rate, wrist movement and sensor coverage in real research excerpts.\n")
    while key or (key := menu()):
        recording = load_recording(key)
        agent = Agent(model, recording)
        print(f"\n{recording.title} | {clock(recording.duration)} | {recording.description}")
        for i, question in enumerate(EXAMPLES, 1):
            print(f"  {i}. {question}")
        while True:
            question = input("\nExample number or question (r = recordings, q = quit): ").strip()
            if question.lower() == "q":
                return
            if question.lower() == "r":
                break
            if question in {"1", "2", "3", "4"}:
                question = EXAMPLES[int(question) - 1]
                print(question)
            print(render(agent.run(question, show_step)))
        key = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", choices=list(catalog()))
    parser.add_argument("--question", help="Run one question and exit")
    parser.add_argument("--json", action="store_true", help="Print full evidence for a single question")
    args = parser.parse_args()
    if args.json and args.question is None:
        parser.error("--json requires --question")
    try:
        model = make_model()
        if args.question is None:
            interactive(model, args.recording)
            return 0
        recording = load_recording(args.recording or "rising")
        if not args.json:
            print(f"Recording: {recording.title}")
        run = Agent(model, recording).run(args.question, None if args.json else show_step)
        print(json.dumps(asdict(run), indent=2) if args.json else render(run))
        return int(run.status == "error")
    except ValueError as error:
        print(f"Configuration or data error: {error}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
