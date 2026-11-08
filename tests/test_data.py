import json

import pytest

from wearable.analysis import signal_quality
from wearable.data import load_recording
from wearable.prepare import movement, prepare, read_e4, timestamp


def test_device_timestamp_formats():
    assert timestamp("1970-01-01 00:00:10") == timestamp("10") == 10
    assert timestamp("1970-01-01T01:00:10+01:00") == 10
    with pytest.raises(ValueError):
        timestamp("NaN")


def test_alignment_and_invalid_raw_row_preserve_clock(tmp_path):
    (tmp_path / "HR.csv").write_text("101\n1\n60\nNaN\n80\n")
    (tmp_path / "ACC.csv").write_text("100,100,100\n2,2,2\n" + "0,0,64\n" * 8)
    rows, metadata = prepare(tmp_path, start=0, duration=4)
    assert [row[1] for row in rows] == ["", 60, "", 80]
    assert [row[2] for row in rows] == [0, 0, 0, 0]
    assert metadata["HR"]["start_offset_s"] == 1
    assert metadata["ACC"]["hz"] == 2


def test_movement_scales_and_removes_constant_gravity():
    assert movement([(0, 0, 64)] * 32) == 0
    assert movement([(-32, 0, 64), (32, 0, 64)] * 16) == 0.5


def test_reject_inconsistent_axis_rates(tmp_path):
    path = tmp_path / "ACC.csv"
    path.write_text("100,100,100\n32,16,32\n0,0,64\n")
    with pytest.raises(ValueError, match="inconsistent"):
        read_e4(path, 3)


def test_loaded_recording_reports_bad_order_gaps_and_missing_channel(tmp_path):
    manifest = {"bad": {"title": "Test fixture", "description": "Synthetic",
                        "duration_s": 6, "signals": {"HR": {"hz": 2}}}}
    (tmp_path / "recordings.json").write_text(json.dumps(manifest))
    (tmp_path / "bad.csv").write_text("second,hr_bpm,hr_samples\n2,65,2\n1,NaN,2\n3,65,1\n"
                                      "2,70,2\n4,80,2\n5,inf,2\n")
    recording = load_recording("bad", tmp_path)
    flags = [issue["kind"] for issue in recording.issues]
    assert flags.count("invalid_measurement") == 3
    assert {"out_of_order", "duplicate_timestamp", "missing_timestamp"} <= set(flags)
    quality = signal_quality(recording)
    assert quality["signals"]["heart_rate"]["missing_s"] == 5
    assert quality["signals"]["heart_rate"]["unexpected_rate"]
    assert not quality["signals"]["accelerometer"]["available"]


def test_real_sensor_offset():
    recording = load_recording("offset")
    assert (recording.hr_hz, recording.acc_hz) == (1, 32)
    assert signal_quality(recording)["signals"]["heart_rate"]["largest_gaps_s"] == [(0, 10)]
    assert recording.seconds[0].motion_g is not None
    assert recording.seconds[10].hr_bpm == 61.0
