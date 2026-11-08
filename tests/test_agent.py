import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APIConnectionError

from wearable.agent import Agent, make_model
from wearable.cli import render
from wearable.data import load_recording


class ScriptedModel:
    """Scripted decisions test the controller, never local model capability."""

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.states = []

    def invoke(self, messages):
        self.states.append(json.loads(messages[-1].content))
        return AIMessage(content=next(self.responses))


def test_action_observation_and_followup():
    model = ScriptedModel('{"action":"heart"}', '{"action":"finish"}',
                          '{"action":"motion","args":{"event":1}}', '{"action":"finish"}')
    agent = Agent(model, load_recording("rising"))
    assert agent.run("Find heart-rate elevations.").status == "ok"
    run = agent.run("How much wrist motion was recorded during the first event?")
    assert [step.action for step in run.steps] == ["motion"]
    assert model.states[1]["results"][0]["observation"]["event_count"] == 1
    assert model.states[2]["previous_heart"]["events"][0]["window_s"] == [465, 960]
    assert "0.0368 g RMS" in render(run)


def test_no_event_prevents_invalid_reference():
    model = ScriptedModel('{"action":"heart"}', '{"action":"motion","args":{"event":1}}', '{"action":"finish"}')
    run = Agent(model, load_recording("steady")).run("Find a rise and check wrist motion.")
    assert run.status == "ok"
    assert [step.action for step in run.steps] == ["heart"]
    assert run.errors == ["The heart result has no event with that ID."]
    assert model.states[-1]["results"][0]["observation"]["events"] == []


@pytest.mark.parametrize("action", ["clarify", "unsupported"])
def test_valid_no_tool_response(action):
    model = ScriptedModel(json.dumps({"action": action}))
    run = Agent(model, load_recording("steady")).run("Some request")
    assert run.status == action and not run.steps


def test_malformed_and_invalid_arguments_are_not_executed():
    model = ScriptedModel("invented text", '{"action":"compare","args":{"a":[0,60],"b":[60,"120"]}}')
    run = Agent(model, load_recording("steady")).run("Compare two intervals.")
    assert run.status == "error"
    assert len(run.errors) == 2 and not run.steps


def test_repeat_and_fabricated_evidence_cannot_finish():
    model = ScriptedModel('{"action":"quality"}', '{"action":"quality"}',
                          '{"action":"finish","args":{"evidence":[5]}}')
    run = Agent(model, load_recording("offset")).run("Check quality.")
    assert run.status == "error"
    assert len(run.steps) == 1 and len(run.errors) == 2


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
