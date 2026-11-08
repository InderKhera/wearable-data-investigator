"""Reproduce aligned demo tables from the original Empatica E4 files."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor, isfinite, sqrt
from pathlib import Path
from statistics import mean, pvariance

from .data import DATA_DIR, FIELDS


@dataclass
class Signal:
    start: float
    hz: int
    samples: list[tuple[float, ...] | None]
    sha256: str


def timestamp(text: str) -> float:
    try:
        result = float(text)
    except ValueError:
        date = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        result = date.timestamp()
    if not isfinite(result):
        raise ValueError("Invalid sensor start timestamp.")
    return result


def read_e4(path: Path, columns: int) -> Signal:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        starts = {timestamp(item) for item in next(reader, [])}
        rates = {float(item) for item in next(reader, [])}
        if len(starts) != 1 or len(rates) != 1:
            raise ValueError(f"{path.name}: inconsistent channel timestamps or rates.")
        start, rate = starts.pop(), rates.pop()
        if not isfinite(rate) or not rate.is_integer() or not 1 <= rate <= 512:
            raise ValueError(f"{path.name}: unsupported sample rate.")
        samples = []
        for row in reader:
            try:
                values = tuple(float(item) for item in row)
                if len(values) != columns or not all(map(isfinite, values)):
                    raise ValueError
                if columns == 1 and values[0] <= 0:
                    raise ValueError
                samples.append(values)
            except ValueError:
                # The row still occupies a sample position in the device clock.
                samples.append(None)
    return Signal(start, int(rate), samples, hashlib.sha256(path.read_bytes()).hexdigest())


def buckets(signal: Signal, origin: float, duration: int) -> dict:
    grouped = defaultdict(list)
    offset = signal.start - origin
    for index, sample in enumerate(signal.samples):
        t = floor(offset + index / signal.hz + 1e-8)
        if 0 <= t < duration and sample is not None:
            grouped[t].append(sample)
    return grouped


def movement(samples: list[tuple[float, ...]]) -> float:
    """RMS deviation from the per-second axis means, in g."""
    return sqrt(sum(pvariance(axis) for axis in zip(*samples))) / 64


def prepare(source: Path, start: int, duration: int):
    signals = {
        key: read_e4(source / f"{key}.csv", columns)
        for key, columns in (("HR", 1), ("ACC", 3))
        if (source / f"{key}.csv").is_file()
    }
    if not signals:
        raise ValueError("Expected HR.csv and/or ACC.csv.")
    if start < 0 or duration <= 0:
        raise ValueError("Start must be nonnegative and duration positive.")
    first = min(signal.start for signal in signals.values())
    origin = first + start
    if origin + duration > max(s.start + len(s.samples) / s.hz for s in signals.values()) + 1e-6:
        raise ValueError("Requested segment extends beyond the recording.")
    groups = {key: buckets(signal, origin, duration) for key, signal in signals.items()}
    rows = []
    for t in range(duration):
        hr = groups.get("HR", {}).get(t, [])
        acc = groups.get("ACC", {}).get(t, [])
        hr_ok = bool(hr) and len(hr) >= 0.8 * signals["HR"].hz
        acc_ok = len(acc) >= 2 and len(acc) >= 0.8 * signals["ACC"].hz
        rows.append((t, round(mean(v[0] for v in hr), 6) if hr_ok else "",
                     round(movement(acc), 8) if acc_ok else "", len(hr), len(acc)))
    metadata = {
        key: {
            "hz": signal.hz,
            "start_offset_s": signal.start - first,
            "source_rows": len(signal.samples),
            "sha256": signal.sha256,
        }
        for key, signal in signals.items()
    }
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("recording", help="Key in data/recordings.json")
    parser.add_argument("--output", type=Path, help="Write regenerated CSV to this path")
    args = parser.parse_args()
    entry = json.loads((DATA_DIR / "recordings.json").read_text())[args.recording]
    rows, metadata = prepare(args.source_dir, entry["start_s"], entry["duration_s"])
    if metadata != entry["signals"]:
        parser.error("Source hashes, rates or offsets differ from the bundled manifest.")
    output = args.output or DATA_DIR / f"{args.recording}.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} seconds to {output}")


if __name__ == "__main__":
    main()
