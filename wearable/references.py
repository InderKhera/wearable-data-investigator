"""Resolve literal elapsed-time references without asking a model to do arithmetic."""
import re

_NUMBER = r"(?:[0-9]{1,5}(?::[0-5][0-9])?)(?![\d:]|\.\d)"
_UNIT = r"minutes?|seconds?|min|sec"
_RANGE = re.compile(
    rf"(?:(?P<prefix>{_UNIT})\s+)?(?<![\w:.+-])(?P<start>{_NUMBER})\s*"
    rf"(?:(?P<start_unit>{_UNIT})\s*)?\s*(?:\bto\b|[–—-])\s*"
    rf"(?:(?P<end_prefix>{_UNIT})\s+)?(?P<end>{_NUMBER})"
    rf"(?:\s*(?P<end_unit>{_UNIT})\b)?", re.IGNORECASE,
)
_CARDINALS = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))
_FIRST = re.compile(
    rf"\bfirst\s+(?P<number>[0-9]{{1,5}}|{'|'.join(_CARDINALS)})\s+(?P<unit>{_UNIT})\b",
    re.IGNORECASE,
)
EVENT_REFERENCE = re.compile(
    r"\b(?:events?|(?:heart[ -]rate|pulse)\s+(?:rises?|increases?|elevations?|peaks?|spikes?)"
    r"|(?:rise|increase|elevation|peak|spike)\s+in\s+(?:heart[ -]rate|pulse)"
    r"|(?:largest|strongest)\s+(?:rise|increase|elevation|peak|spike))\b", re.IGNORECASE,
)
EVENT_ID = re.compile(r"\bevent\s*#?([0-9]{1,5})\b", re.IGNORECASE)
TIME_ORDER_REFERENCE = re.compile(r"\b(earlier|later)\s+(?:heart[ -]rate\s+)?event\b")
TOP_EVENT = re.compile(r"\b(?:largest|strongest|biggest|first event)\b", re.IGNORECASE)
BASELINE_REFERENCE = re.compile(r"\b(?:baseline|reference period)\b", re.IGNORECASE)


def _seconds(value: str, unit: str) -> int:
    if ":" in value:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)
    return int(value) * (60 if unit.startswith("m") else 1)


def explicit_windows(question: str) -> list[list[int]]:
    """Read explicit ranges with seconds/minutes units or MM:SS clocks."""
    found = []
    for match in _RANGE.finditer(question):
        start, end = match["start"], match["end"]
        units = [match[key].lower() for key in ("prefix", "start_unit", "end_prefix", "end_unit")
                 if match[key]]
        if ":" in start and ":" in end and not units:
            found.append((match.start(), [_seconds(start, ""), _seconds(end, "")]))
        elif units and ":" not in start and ":" not in end and len({unit[0] for unit in units}) == 1:
            found.append((match.start(), [_seconds(start, units[0]), _seconds(end, units[0])]))
    for match in _FIRST.finditer(question):
        number = match["number"].lower()
        if number in _CARDINALS:
            number = str(_CARDINALS[number])
        found.append((match.start(), [0, _seconds(number, match["unit"].lower())]))
    return [window for _, window in sorted(found)]
