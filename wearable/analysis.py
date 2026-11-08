"""Deterministic investigations on the shared one-second recording grid."""
from itertools import groupby
from statistics import mean, median, pstdev

from .data import Recording

BASELINE_S = 120
ELEVATION_BPM = 10
MIN_EVENT_S = 20
MIN_COVERAGE = 0.8
MIN_STATS_S = 10
Window = tuple[int, int]


def validate_window(recording: Recording, window: Window):
    start, end = window
    if not 0 <= start < end <= recording.duration:
        raise ValueError(f"Use an interval within [0, {recording.duration}) seconds.")


def runs(flags):
    """Yield half-open intervals for contiguous true values."""
    start = 0
    for flag, group in groupby(flags):
        end = start + sum(1 for _ in group)
        if flag:
            yield start, end
        start = end


def stats(recording: Recording, window: Window, field: str):
    validate_window(recording, window)
    start, end = window
    values = [getattr(row, field) for row in recording.seconds[start:end]]
    valid = [value for value in values if value is not None]
    coverage = len(valid) / len(values)
    info = {"window_s": list(window), "valid_s": len(valid), "total_s": len(values),
            "coverage": round(coverage, 4)}
    if len(valid) < MIN_STATS_S or coverage < MIN_COVERAGE:
        return {**info, "status": "insufficient"}
    return {**info, "status": "ok", "mean": round(mean(valid), 6),
            "median": round(median(valid), 6), "sd": round(pstdev(valid), 6),
            "min": round(min(valid), 6), "max": round(max(valid), 6)}


def signal_quality(recording: Recording):
    report = {}
    for name, field, count, rate, expected in (
        ("heart_rate", "hr_bpm", "hr_samples", recording.hr_hz, 1),
        ("accelerometer", "motion_g", "acc_samples", recording.acc_hz, 32),
    ):
        invalid = [getattr(row, field) is None for row in recording.seconds]
        gaps = list(runs(invalid))
        report[name] = {
            "available": rate is not None,
            "sample_rate_hz": rate,
            "expected_e4_rate_hz": expected,
            "unexpected_rate": rate is not None and rate != expected,
            "missing_s": sum(invalid),
            "partial_bins": sum(0 < getattr(row, count) < rate for row in recording.seconds) if rate else 0,
            "coverage": round(1 - sum(invalid) / recording.duration, 4),
            "gap_count": len(gaps),
            "largest_gaps_s": sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:4],
        }
    return {"duration_s": recording.duration, "signals": report,
            "issue_count": len(recording.issues), "examples": list(recording.issues[:5])}


def heart_rate_events(recording: Recording):
    overall = stats(recording, (0, recording.duration), "hr_bpm")
    first = stats(recording, (0, min(60, recording.duration)), "hr_bpm")
    last = stats(recording, (max(0, recording.duration - 60), recording.duration), "hr_bpm")
    trend = round(last["mean"] - first["mean"], 3) if (
        recording.duration >= 120 and first["status"] == last["status"] == "ok"
    ) else None
    result = {"overall": overall, "last_vs_first_minute_bpm": trend, "events": [], "event_count": 0}
    if recording.duration < BASELINE_S + MIN_EVENT_S:
        return {**result, "status": "insufficient", "reason": "Need 120 baseline seconds and 20 later seconds."}
    baseline = stats(recording, (0, BASELINE_S), "hr_bpm")
    later = stats(recording, (BASELINE_S, recording.duration), "hr_bpm")
    if baseline["status"] != "ok" or later["status"] != "ok":
        return {**result, "status": "insufficient", "reason": "Baseline or later heart-rate coverage is below 80%."}
    reference = baseline["median"]
    threshold = reference + ELEVATION_BPM
    elevated = [row.hr_bpm is not None and row.hr_bpm >= threshold
                for row in recording.seconds[BASELINE_S:]]
    events = []
    for start, end in runs(elevated):
        if end - start < MIN_EVENT_S:
            continue
        window = (start + BASELINE_S, end + BASELINE_S)
        event = stats(recording, window, "hr_bpm")
        events.append({"window_s": list(window), "peak_bpm": event["max"],
                       "mean_bpm": event["mean"], "peak_above_baseline_bpm": round(event["max"] - reference, 3)})
    events.sort(key=lambda event: (-event["peak_bpm"], event["window_s"][0]))
    return {**result, "status": "ok", "baseline": baseline, "threshold_bpm": threshold,
            "minimum_duration_s": MIN_EVENT_S, "event_count": len(events),
            "events": [{"id": index, **event} for index, event in enumerate(events[:3], 1)]}


def summarize_window(recording: Recording, window: Window):
    return {"window_s": list(window), "heart_rate_bpm": stats(recording, window, "hr_bpm"),
            "wrist_motion_g": stats(recording, window, "motion_g")}


def delta(a: dict, b: dict, field: str):
    before, after = a[field], b[field]
    return round(after["mean"] - before["mean"], 6) if before["status"] == after["status"] == "ok" else None


def compare_intervals(recording: Recording, a: Window, b: Window):
    left, right = summarize_window(recording, a), summarize_window(recording, b)
    return {"a": left, "b": right, "b_minus_a_bpm": delta(left, right, "heart_rate_bpm"),
            "b_minus_a_motion_g": delta(left, right, "wrist_motion_g")}


def activity_context(recording: Recording, window: Window):
    start, end = window
    during = summarize_window(recording, window)
    span = end - start
    before = summarize_window(recording, (max(0, start - span), start)) if start else None
    after = summarize_window(recording, (end, min(recording.duration, end + span))) if end < recording.duration else None
    difference = delta(before, during, "wrist_motion_g") if before else None
    change = None
    if difference is not None:
        threshold = max(0.01, 0.25 * before["wrist_motion_g"]["mean"])
        change = "higher" if difference >= threshold else "lower" if difference <= -threshold else "similar"
    return {"before": before, "during": during, "after": after,
            "during_minus_before_motion_g": difference, "relative_motion": change,
            "note": "Wrist acceleration measures local motion. Timing alone cannot establish a cause."}
