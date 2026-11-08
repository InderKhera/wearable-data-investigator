"""Select a wearable analysis goal with LangChain and execute its bounded tools."""
import json
import os
from dataclasses import dataclass, field
from time import perf_counter
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import analysis as analytics
from .data import Recording
from .references import (
    BASELINE_REFERENCE, EVENT_ID, EVENT_REFERENCE, TIME_ORDER_REFERENCE, TOP_EVENT, explicit_windows,
)

DEFAULT_MODEL = "qwen2.5-0.5b-instruct-q4_k_m"
DEFAULT_URL = "http://127.0.0.1:8083/mimik-ai/openai/v1"
MAX_TOOL_CALLS = 2

ROUTE_PROMPT = """Classify the user's wearable analysis request. Reply with one label.
quality = check recording coverage, sensor availability or missing measurements
heart = find or summarize heart-rate events
motion = measure wrist movement, including during a heart-rate event
compare = contrast two time windows or compare an event with a baseline
unsupported = diagnose a condition, give medical advice, or an unrelated request
When the user wants both heart rate and movement, choose motion. When the user wants a comparison, choose compare.
"""
ROUTE_EXAMPLES = (
    ("Is this recording complete?", "quality"),
    ("When did pulse stay elevated?", "heart"),
    ("Was there wrist motion during the pulse spike?", "motion"),
    ("Compare minutes 2–3 and 4–5.", "compare"),
    ("Does this indicate heart disease?", "unsupported"),
)

EVENT_CHOICE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Match a requested heart-rate event to a candidate. Reply with one event ID or clarify. "
               "If the request gives no reference, reply clarify."),
    ("human", "Request: What was movement like at the earlier event?\nEarlier event: 3\nLater event: 2"),
    ("ai", "3"),
    ("human", "Request: Check wrist motion at the later event.\nEarlier event: 1\nLater event: 3"),
    ("ai", "3"),
    ("human", "Request: Check movement at an event.\nEarlier event: 1\nLater event: 2"),
    ("ai", "clarify"),
    ("human", "Request: {question}\nEarlier event: {earlier}\nLater event: {later}"),
])


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


@dataclass
class Step:
    action: str
    args: dict
    result: dict
    selected_by: str = "model_goal"


@dataclass
class Run:
    status: str
    steps: list[Step]
    shown: list[Step]
    message: str = ""
    errors: list[str] = field(default_factory=list)
    model_calls: int = 0
    elapsed_s: float = 0.0
    goal: str | None = None
    stop_reason: str = ""
    safeguards: list[str] = field(default_factory=list)


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
                      temperature=0, extra_body={"max_tokens": 12}, timeout=25, max_retries=0,
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

    def run(self, question: str, on_step=None, on_response=None) -> Run:
        start = perf_counter()
        steps: list[Step] = []
        errors: list[str] = []
        safeguards: list[str] = []
        seen = set()
        calls = 0
        goal = None

        def result(status, reason, message="", shown=None):
            return Run(status, steps, steps if shown is None else shown, message, errors,
                       calls, round(perf_counter() - start, 3), goal, reason, safeguards)

        def choose(messages, labels):
            nonlocal calls
            calls += 1
            with tracing_context(enabled=False):
                answer = self.model.invoke(messages)
            if on_response is not None:
                on_response(answer)
            label = answer.content.strip().lower() if isinstance(answer.content, str) else ""
            if label not in labels:
                errors.append("Model reply was not an allowed label.")
                return None
            return labels[label]

        def execute(name, args, selected_by="model_goal"):
            tool = self.tools[name]
            args = tool.args_schema.model_validate(args).model_dump(exclude_none=True)
            signature = (name, json.dumps(args, sort_keys=True))
            if signature in seen or len(steps) >= MAX_TOOL_CALLS:
                raise ValueError("Duplicate call or tool budget exceeded.")
            output = tool.invoke(args)
            seen.add(signature)
            step = Step(name, args, output, selected_by)
            steps.append(step)
            if name == "heart":
                self.last_heart = step
            if on_step is not None:
                on_step(tool.name, step)

        if not question.strip() or len(question) > 1500:
            return result("clarify", "invalid_question", "Enter a question between 1 and 1500 characters.")
        try:
            messages = [SystemMessage(ROUTE_PROMPT)]
            for sample, label in ROUTE_EXAMPLES:
                messages.extend((HumanMessage(sample), AIMessage(label)))
            messages.append(HumanMessage(question))
            labels = {name: name for name in self.tools}
            labels.update(unsupported="unsupported", diagnose="unsupported")
            goal = choose(messages, labels)
            if goal is None:
                return result("error", "invalid_goal", "The model did not select a supported investigation.")
            if goal == "unsupported":
                return result("unsupported", "model_declined",
                              "I can investigate recorded heart rate, wrist motion and signal quality.")
            if goal in {"quality", "heart"}:
                execute(goal, {})
                return result("ok", "goal_satisfied")

            windows = explicit_windows(question)
            if goal == "compare" and len(windows) == 2:
                execute(goal, {"a": windows[0], "b": windows[1]})
                return result("ok", "goal_satisfied")
            if goal == "motion" and len(windows) == 1:
                execute(goal, {"window": windows[0]})
                return result("ok", "goal_satisfied")

            needs_event = EVENT_REFERENCE.search(question) and (
                (goal == "motion" and not windows) or
                (goal == "compare" and (len(windows) == 1 or
                 (not windows and BASELINE_REFERENCE.search(question))))
            )
            if not needs_event:
                safeguards.append("missing_reference")
                return result("clarify", "missing_reference",
                              "Specify the elapsed time window(s) or a heart-rate event and reference period.")
            if self.last_heart is None:
                execute("heart", {}, "prerequisite")
            evidence = self.last_heart
            events = evidence.result["events"]
            if not events:
                safeguards.append("event_followup_unavailable")
                return result("ok", "no_event_evidence", shown=[evidence])

            explicit_id = EVENT_ID.search(question)
            if explicit_id:
                event = int(explicit_id[1])
            elif len(events) == 1 or TOP_EVENT.search(question):
                event = events[0]["id"]
            else:
                event = None
                references = TIME_ORDER_REFERENCE.findall(question.lower())
                if len(events) == 2 and len(references) == 1:
                    earlier, later = sorted(events, key=lambda item: item["window_s"][0])
                    ids = (earlier["id"], later["id"])
                    labels = {str(event_id): event_id for event_id in ids}
                    labels["clarify"] = None
                    event = choose(EVENT_CHOICE_PROMPT.format_messages(
                        question=question, earlier=ids[0], later=ids[1],
                    ), labels)
                    expected_id = ids[0] if references[0] == "earlier" else ids[1]
                    if event is not None and event != expected_id:
                        safeguards.append("event_choice_conflicts_with_reference")
                        event = None
            if event not in {item["id"] for item in events}:
                safeguards.append("unresolved_event")
                return result("clarify", "unresolved_event", "Which observed event should I use?",
                              shown=[evidence])
            args = {"event": event}
            if goal == "compare" and windows and windows[0] != [0, analytics.BASELINE_S]:
                args = {"a": windows[0], "b": list(self._event_window(event))}
            execute(goal, args)
            return result("ok", "goal_satisfied")
        except (APIConnectionError, APIStatusError) as error:
            return result("error", "inference_error", model_error(error))
        except (ValidationError, ValueError) as error:
            safeguards.append("invalid_tool_arguments")
            errors.append(str(error))
            return result("clarify", "invalid_tool_arguments", "Provide valid elapsed windows within the recording.")
