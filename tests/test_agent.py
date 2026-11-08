import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APIConnectionError

from wearable.agent import Agent, make_model
from wearable.cli import render
from wearable.data import load_recording


class ScriptedModel:
    """Scripted choices test the controller, never local model capability."""

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages)
        return AIMessage(content=next(self.responses))


def test_action_observation_and_followup():
    model = ScriptedModel("heart", "motion")
    agent = Agent(model, load_recording("rising"))
    assert agent.run("Find heart-rate elevations.").stop_reason == "goal_satisfied"
    run = agent.run("How much wrist motion was recorded during the first event?")
    assert [step.action for step in run.steps] == ["motion"]
    assert run.steps[0].args == {"event": 1}
    assert run.steps[0].result["during"]["window_s"] == [465, 960]
    assert "0.0368 g RMS" in render(run)
    assert run.model_calls == 1


def test_no_event_prevents_invalid_reference():
    question = "Find a rise in heart rate and check wrist motion."
    steady = Agent(ScriptedModel("motion"), load_recording("steady")).run(question)
    rising = Agent(ScriptedModel("motion"), load_recording("rising")).run(question)
    assert steady.status == rising.status == "ok"
    assert steady.stop_reason == "no_event_evidence"
    assert steady.safeguards == ["event_followup_unavailable"]
    assert [step.action for step in steady.steps] == ["heart"]
    assert [step.action for step in rising.steps] == ["heart", "motion"]
    assert [step.selected_by for step in rising.steps] == ["prerequisite", "model_goal"]
    assert rising.steps[-1].args == {"event": 1}
    assert steady.model_calls == rising.model_calls == 1


@pytest.mark.parametrize("label", ["unsupported", "diagnose"])
def test_valid_no_tool_response(label):
    run = Agent(ScriptedModel(label), load_recording("steady")).run("Can you diagnose this?")
    assert run.status == "unsupported" and not run.steps
    assert run.stop_reason == "model_declined"


def test_malformed_and_invalid_arguments_are_not_executed():
    rec = load_recording("steady")
    invalid = Agent(ScriptedModel('{"action":"compare","a":[0,60],"b":[60,"120"]}'), rec)
    run = invalid.run("Compare two intervals.")
    assert run.status == "error" and run.stop_reason == "invalid_goal"
    assert not run.steps

    out_of_bounds = Agent(ScriptedModel("compare"), rec)
    run = out_of_bounds.run("Compare seconds 0 to 60 and 60 to 1000 seconds.")
    assert run.status == "clarify" and not run.steps
    assert run.safeguards == ["invalid_tool_arguments"]


def test_goal_completion_needs_no_repeat_or_model_supplied_evidence():
    model = ScriptedModel("quality")
    run = Agent(model, load_recording("offset")).run("Check quality.")
    assert run.status == "ok" and run.stop_reason == "goal_satisfied"
    assert len(run.steps) == run.model_calls == 1
    assert run.shown == run.steps
    forged = Agent(ScriptedModel('{"action":"finish","evidence":[5]}'), load_recording("offset"))
    assert forged.run("Check quality.").stop_reason == "invalid_goal"


def test_clarifies_missing_and_unobserved_references():
    rec = load_recording("rising")
    run = Agent(ScriptedModel("compare"), rec).run("Compare two time windows.")
    assert run.status == "clarify" and run.stop_reason == "missing_reference"
    assert not run.steps
    run = Agent(ScriptedModel("motion"), rec).run("Check movement during event 99.")
    assert run.status == "clarify" and run.stop_reason == "unresolved_event"
    assert [step.action for step in run.steps] == ["heart"]


def test_model_can_choose_between_observed_events():
    from wearable.data import Recording, Second

    values = [60] * 120 + [80] * 30 + [60] * 20 + [90] * 30
    rec = Recording("test", "Synthetic", "", tuple(Second(hr, 0.03, 1, 32) for hr in values), 1, 32)
    model = ScriptedModel("motion", "2")
    run = Agent(model, rec).run("Check wrist movement at the earlier event.")
    assert run.status == "ok" and run.model_calls == 2
    assert "Earlier event: 2\nLater event: 1" in model.requests[-1][-1].content
    assert run.steps[-1].args == {"event": 2}
    assert run.steps[-1].result["during"]["window_s"] == [120, 150]

    wrong_id = Agent(ScriptedModel("motion", "1"), rec).run("Check wrist movement at the earlier event.")
    assert wrong_id.status == "clarify"
    assert "event_choice_conflicts_with_reference" in wrong_id.safeguards
    assert [step.action for step in wrong_id.steps] == ["heart"]

    unclear = Agent(ScriptedModel("motion"), rec).run("Check wrist movement at an event.")
    assert unclear.status == "clarify"
    assert unclear.model_calls == 1
    assert [step.action for step in unclear.steps] == ["heart"]


def test_endpoint_connection_failure():
    class Unavailable:
        def invoke(self, messages):
            raise APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8083"))
    run = Agent(Unavailable(), load_recording("steady")).run("Check quality.")
    assert run.status == "error"
    assert "Cannot reach mimOE" in run.message


def test_model_configuration_rejects_nonlocal_endpoint(monkeypatch):
    monkeypatch.setenv("MIMOE_BASE_URL", "https://example.com/v1")
    with pytest.raises(ValueError, match="loopback"):
        make_model()


def test_explicit_interval_units_and_unusable_ranges():
    from wearable.references import explicit_windows

    assert explicit_windows("minute 0 to 1 with minute 1 to 2") == [[0, 60], [60, 120]]
    assert explicit_windows("00:40–02:00 and 180 to 240 seconds") == [[40, 120], [180, 240]]
    assert explicit_windows("0-60 seconds and 60-120 seconds") == [[0, 60], [60, 120]]
    assert explicit_windows("from 01:00 to 02:00.") == [[60, 120]]
    assert explicit_windows("first two minutes") == [[0, 120]]
    assert explicit_windows("-5 to 10 seconds") == []
    assert explicit_windows("0.5 to 1.5 minutes") == []
    assert explicit_windows("00:75 to 02:00") == []


def test_langchain_sends_mimoe_completion_cap(monkeypatch):
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        assert request.url.path == "/mimik-ai/openai/v1/chat/completions"
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
        })

    monkeypatch.setenv("MIMOE_BASE_URL", "http://127.0.0.1:8083/mimik-ai/openai/v1")
    monkeypatch.setenv("MIMOE_API_KEY", "test-token")
    monkeypatch.setenv("MIMOE_MODEL", "test-model")
    class LocalClient(httpx.Client):
        def __init__(self, **options):
            super().__init__(transport=httpx.MockTransport(respond), **options)

    monkeypatch.setattr("wearable.agent.httpx.Client", LocalClient)
    answer = make_model().invoke("Check the recording.")
    assert answer.response_metadata["model_name"] == "test-model"
    assert sent[0]["max_tokens"] == 12
    assert "max_completion_tokens" not in sent[0]
