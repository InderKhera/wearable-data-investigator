import pytest

from wearable.analysis import activity_context, compare_intervals, heart_rate_events
from wearable.data import Recording, Second, load_recording


def synthetic(heart_rates, motion=None):
    motion = motion if motion is not None else [0.02] * len(heart_rates)
    rows = tuple(Second(hr, acc, int(hr is not None), 32 if acc is not None else 0)
                 for hr, acc in zip(heart_rates, motion))
    return Recording("synthetic", "Synthetic fixture", "", rows, 1, 32)


def test_event_requires_duration_and_gaps_break_streak():
    rec = synthetic([60] * 120 + [80] * 12 + [None] + [80] * 12 + [65] + [72] * 20)
    result = heart_rate_events(rec)
    assert result["baseline"]["median"] == 60
    assert result["threshold_bpm"] == 70
    assert [event["window_s"] for event in result["events"]] == [[146, 166]]


@pytest.mark.parametrize("rates", [[60] * 50, [None] * 30 + [60] * 90 + [80] * 50])
def test_insufficient_baseline(rates):
    assert heart_rate_events(synthetic(rates))["status"] == "insufficient"


def test_before_during_after_uses_observed_windows():
    rec = synthetic([60] * 20 + [80] * 20 + [65] * 20, [0.02] * 20 + [0.12] * 20 + [0.01] * 20)
    result = activity_context(rec, (20, 40))
    assert result["before"]["window_s"] == [0, 20]
    assert result["after"]["window_s"] == [40, 60]
    assert result["during"]["heart_rate_bpm"]["mean"] == 80
    assert result["during_minus_before_motion_g"] == pytest.approx(0.1)
    assert result["relative_motion"] == "higher"


def test_interval_comparison_does_not_fill_missing_signal():
    rec = synthetic([60] * 30 + [70] * 30, [0.1] * 30 + [None] * 10 + [0.2] * 20)
    result = compare_intervals(rec, (0, 30), (30, 60))
    assert result["b_minus_a_bpm"] == 10
    assert result["a"]["heart_rate_bpm"]["sd"] == 0
    assert result["b"]["wrist_motion_g"]["valid_s"] == 20
    assert result["b_minus_a_motion_g"] is None
    with pytest.raises(ValueError, match="interval"):
        compare_intervals(rec, (0, 30), (30, 61))


def test_bundled_recordings_yield_distinct_observations():
    assert heart_rate_events(load_recording("steady"))["events"] == []
    rec = load_recording("rising")
    event = heart_rate_events(rec)["events"][0]
    assert event["window_s"] == [465, 960]
    assert event["peak_bpm"] == pytest.approx(128.27)
    assert activity_context(rec, tuple(event["window_s"]))["during_minus_before_motion_g"] == pytest.approx(0.036809)
