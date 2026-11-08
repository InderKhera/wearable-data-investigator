"""Load the one-second analysis grid and retain evidence of input problems."""
import csv
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIELDS = ["second", "hr_bpm", "motion_g", "hr_samples", "acc_samples"]


@dataclass(frozen=True)
class Second:
    hr_bpm: float | None = None
    motion_g: float | None = None
    hr_samples: int = 0
    acc_samples: int = 0


@dataclass(frozen=True)
class Recording:
    key: str
    title: str
    description: str
    seconds: tuple[Second, ...]
    hr_hz: int | None
    acc_hz: int | None
    issues: tuple[dict, ...] = ()

    @property
    def duration(self) -> int:
        return len(self.seconds)


def catalog(directory: Path = DATA_DIR) -> dict:
    return json.loads((directory / "recordings.json").read_text(encoding="utf-8"))


def load_recording(key: str, directory: Path = DATA_DIR) -> Recording:
    entry = catalog(directory)[key]
    duration = entry["duration_s"]
    hr_hz = entry["signals"].get("HR", {}).get("hz")
    acc_hz = entry["signals"].get("ACC", {}).get("hz")
    if type(duration) is not int or not 1 <= duration <= 86_400:
        raise ValueError("Recording duration must be 1 to 86400 whole seconds.")
    for hz in (hr_hz, acc_hz):
        if hz is not None and (type(hz) is not int or not 1 <= hz <= 512):
            raise ValueError("Sample rate must be a positive integer up to 512 Hz.")

    def read_measurement(row: dict, name: str, count_name: str, rate: int | None, t: int):
        raw = row.get(name)
        try:
            count = int(row.get(count_name) or 0)
            value = float(raw) if raw else None
            if not 0 <= count <= (rate or 0):
                raise ValueError
            if value is not None and (
                not isfinite(value) or value < 0 or (name == "hr_bpm" and value == 0)
                or count < 0.8 * (rate or 1)
            ):
                raise ValueError
            return value, count
        except ValueError:
            issues.append({"kind": "invalid_measurement", "signal": name, "second": t})
            return None, 0

    rows = [Second() for _ in range(duration)]
    seen = set()
    previous = -1
    issues = []
    with (directory / f"{key}.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "second" not in reader.fieldnames:
            raise ValueError("Recording CSV needs a second column.")
        for row in reader:
            try:
                t = int(row.get("second") or "")
                if not 0 <= t < duration:
                    raise ValueError
            except ValueError:
                issues.append({"kind": "invalid_timestamp", "row": reader.line_num})
                continue
            if t in seen:
                issues.append({"kind": "duplicate_timestamp", "second": t})
                rows[t] = Second()
                continue
            if t < previous:
                issues.append({"kind": "out_of_order", "second": t})
            previous = t
            seen.add(t)
            hr, hr_n = read_measurement(row, "hr_bpm", "hr_samples", hr_hz, t)
            acc, acc_n = read_measurement(row, "motion_g", "acc_samples", acc_hz, t)
            rows[t] = Second(hr, acc, hr_n, acc_n)
    for t in sorted(set(range(duration)) - seen):
        issues.append({"kind": "missing_timestamp", "second": t})
    return Recording(key, entry["title"], entry["description"], tuple(rows), hr_hz, acc_hz, tuple(issues))
