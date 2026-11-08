"""A bounded LangChain action loop over read-only wearable analysis tools."""
import json
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal
from urllib.parse import urlparse

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import analysis as analytics
from .data import Recording

DEFAULT_MODEL = "SmolLM2-360M-Instruct-Q2_K"
DEFAULT_URL = "http://127.0.0.1:8083/mimik-ai/openai/v1"
MAX_STEPS = 6

PROMPT = """You operate a wearable data investigator. Reply with one JSON action and nothing else.
Actions:
{"action":"quality"} checks sensor coverage and missing data.
{"action":"heart"} finds sustained heart-rate elevations and summarizes heart rate.
{"action":"motion","args":{"event":1}} measures movement around event 1 from heart results.
{"action":"motion","args":{"window":[60,120]}} measures movement at explicit elapsed seconds.
{"action":"compare","args":{"a":[0,60],"b":[60,120]}} compares explicit elapsed windows.
{"action":"compare","args":{"event":1}} compares event 1 to the first 120 seconds.
{"action":"finish"} displays the current tool results.
{"action":"clarify"} asks for an event or elapsed time window.
{"action":"unsupported"} handles requests outside wearable measurements, including diagnosis and advice.
Use half-open seconds, converting minutes to seconds. Stay within the recording.
If asked about movement or comparison at the biggest heart-rate rise, call heart first unless a heart result is already available. Then choose motion or compare as requested. Use event 1 for the biggest event. If heart found no events, finish with that result. Never invent an event ID.
For an isolated quality or heart question, use that tool and then finish. For explicit windows, compare directly. A tool error is an observation you may correct. Never repeat a successful call in the same turn.
Example:
Question: Was the strongest heart-rate rise accompanied by wrist motion?
With no evidence: {"action":"heart"}
After heart found event 1: {"action":"motion","args":{"event":1}}
After motion, or if heart found no events: {"action":"finish"}
"""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Empty(StrictModel):
    pass


class MotionArgs(StrictModel):
    event: int | None = Field(default=None, ge=1, le=3)
    window: list[int] | None = Field(default=None, min_length=2, max_length=2)


class CompareArgs(StrictModel):
    event: int | None = Field(default=None, ge=1, le=3)
    a: list[int] | None = Field(default=None, min_length=2, max_length=2)
    b: list[int] | None = Field(default=None, min_length=2, max_length=2)


class Action(StrictModel):
    action: Literal["quality", "heart", "motion", "compare", "finish", "clarify", "unsupported"]
    args: dict = Field(default_factory=dict)


class FinishArgs(StrictModel):
    evidence: list[int] | None = Field(default=None, min_length=1, max_length=5)


@dataclass
class Step:
    action: str
    args: dict
    result: dict


@dataclass
class Run:
    status: str
    steps: list[Step]
    shown: list[Step]
    message: str = ""
    errors: list[str] = field(default_factory=list)
    model_calls: int = 0
    elapsed_s: float = 0.0


def make_model() -> ChatOpenAI:
    base = os.getenv("MIMOE_BASE_URL", DEFAULT_URL).rstrip("/")
    parsed = urlparse(base)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("MIMOE_BASE_URL must point to a loopback HTTP(S) endpoint.")
    if base.endswith("/chat/completions"):
        raise ValueError("Use the base URL ending in /v1, without /chat/completions.")
    token = os.getenv("MIMOE_API_KEY")
    if not token:
        raise ValueError("Set MIMOE_API_KEY to the token shown by Studio.")
    return ChatOpenAI(base_url=base, api_key=token, model=os.getenv("MIMOE_MODEL", DEFAULT_MODEL),
                      temperature=0, max_tokens=192, timeout=25, max_retries=0,
                      use_responses_api=False, http_client=httpx.Client(trust_env=False))


def model_error(error: Exception) -> str:
    if isinstance(error, APITimeoutError):
        return "mimOE timed out. Check that the selected model is loaded, then retry."
    if isinstance(error, APIConnectionError):
        return "Cannot reach mimOE. Start the local runtime and verify MIMOE_BASE_URL."
    if error.status_code in {401, 403}:
        return "mimOE rejected authentication. Check MIMOE_API_KEY in Studio."
    if error.status_code == 404:
        return "mimOE returned HTTP 404. Verify the model ID and base URL in Studio."
    return f"mimOE returned HTTP {error.status_code}. Check the selected model in Studio."


def parse_action(content: str) -> Action:
    content = content.strip()
    if content.startswith("```json\n") and content.endswith("\n```"):
        content = content[8:-4]
    return Action.model_validate_json(content)


def compact(step: Step) -> dict:
    """Return a small observation to the model. Full evidence stays in Python."""
    if step.action == "heart":
        return {key: step.result[key] for key in ("status", "event_count", "events")}
    if step.action == "quality":
        return {name: {"available": signal["available"], "missing_s": signal["missing_s"]}
                for name, signal in step.result["signals"].items()}
    if step.action == "motion":
        return {"relative_motion": step.result["relative_motion"],
                "during_s": step.result["during"]["window_s"]}
    return {key: step.result[key] for key in ("b_minus_a_bpm", "b_minus_a_motion_g")}


class Agent:
    def __init__(self, model, recording: Recording):
        self.model = model
        self.recording = recording
        self.last_heart: Step | None = None
        self.tools = {
            "quality": StructuredTool.from_function(
                name="inspect_signal_quality", description="Check coverage, timestamps and rates.",
                func=lambda: analytics.signal_quality(recording), args_schema=Empty),
            "heart": StructuredTool.from_function(
                name="find_heart_rate_events", description="Find sustained elevations relative to baseline.",
                func=lambda: analytics.heart_rate_events(recording), args_schema=Empty),
            "motion": StructuredTool.from_function(
                name="analyze_activity_context", description="Measure motion before, during and after a window.",
                func=self._motion, args_schema=MotionArgs),
            "compare": StructuredTool.from_function(
                name="compare_intervals", description="Compare two windows in the selected recording.",
                func=self._compare, args_schema=CompareArgs),
        }

    def _event_window(self, event: int) -> analytics.Window:
        if self.last_heart is None:
            raise ValueError("Run heart before referencing an event.")
        for candidate in self.last_heart.result["events"]:
            if candidate["id"] == event:
                return tuple(candidate["window_s"])
        raise ValueError("The heart result has no event with that ID.")

    def _motion(self, event: int | None = None, window: list[int] | None = None):
        if (event is None) == (window is None):
            raise ValueError("Give one observed event ID or one explicit window.")
        interval = self._event_window(event) if event is not None else tuple(window)
        return analytics.activity_context(self.recording, interval)

    def _compare(self, event: int | None = None, a: list[int] | None = None, b: list[int] | None = None):
        if event is not None and a is None and b is None:
            return analytics.compare_intervals(self.recording, (0, analytics.BASELINE_S), self._event_window(event))
        if event is None and a is not None and b is not None:
            return analytics.compare_intervals(self.recording, tuple(a), tuple(b))
        raise ValueError("Give one event ID or both windows a and b.")

    def run(self, question: str, on_step=None) -> Run:
        start = perf_counter()
        steps: list[Step] = []
        errors: list[str] = []
        seen = set()
        calls = 0

        def result(status, message="", shown=None):
            return Run(status, steps, steps if shown is None else shown, message, errors,
                       calls, round(perf_counter() - start, 3))

        if not question.strip() or len(question) > 1500:
            return result("clarify", "Enter a question between 1 and 1500 characters.")
        for _ in range(MAX_STEPS):
            state = {
                "duration_s": self.recording.duration,
                "question": question,
                "previous_heart": compact(self.last_heart) if self.last_heart and
                not any(step.action == "heart" for step in steps) else None,
                "results": [{"id": i, "tool": step.action, "observation": compact(step)}
                            for i, step in enumerate(steps, 1)],
                "last_error": errors[-1] if errors else None,
            }
            try:
                calls += 1
                with tracing_context(enabled=False):
                    message = self.model.invoke([SystemMessage(content=PROMPT),
                                                 HumanMessage(content=json.dumps(state))])
            except (APIConnectionError, APIStatusError) as error:
                return result("error", model_error(error))
            try:
                if not isinstance(message.content, str):
                    raise ValueError("Expected a text action.")
                action = parse_action(message.content)
                if action.action == "finish":
                    requested = FinishArgs.model_validate(action.args).evidence
                    if not steps:
                        raise ValueError("Run at least one tool before finishing.")
                    indices = requested or list(range(1, len(steps) + 1))
                    if len(set(indices)) != len(indices) or any(not 1 <= i <= len(steps) for i in indices):
                        raise ValueError("Evidence must contain distinct IDs from this turn.")
                    return result("ok", shown=[steps[i - 1] for i in indices])
                if action.action in {"clarify", "unsupported"}:
                    Empty.model_validate(action.args)
                    message = ("Which elapsed time window or detected event should I use?"
                               if action.action == "clarify" else
                               "I can investigate recorded heart rate, wrist motion and signal quality.")
                    return result(action.action, message)
                tool = self.tools[action.action]
                args = tool.args_schema.model_validate(action.args).model_dump(exclude_none=True)
                signature = (action.action, json.dumps(args, sort_keys=True))
                if signature in seen:
                    raise ValueError("This tool already ran with those arguments. Use its result.")
                output = tool.invoke(args)
                seen.add(signature)
            except (ValidationError, ValueError) as error:
                if isinstance(error, ValidationError):
                    issue = error.errors()[0]
                    reason = f"{'.'.join(map(str, issue['loc'])) or 'action'}: {issue['msg']}"
                    # JSON parse errors may contain snippets of the model output.
                    if issue["type"] == "json_invalid":
                        reason = "Expected one complete JSON action."
                else:
                    reason = str(error)
                errors.append(reason)
                if len(errors) >= 2:
                    return result("error", "The local model failed to produce a valid next action after one repair.")
                continue
            step = Step(action.action, args, output)
            steps.append(step)
            if action.action == "heart":
                self.last_heart = step
            if on_step is not None:
                on_step(tool.name, step)
        return result("error", "The investigation reached its action limit. Any available evidence is shown below.")
