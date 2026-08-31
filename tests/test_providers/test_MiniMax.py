"""Tests for the MiniMax connector.

These tests mock the HTTP layer (via the ``http_post=``
constructor arg) and do not require network access.

Pinned contract:

- The library never sends ``temperature`` / ``top_p`` /
  ``max_p`` / ``seed`` / ``penalties`` unless the provider
  API requires one of them. The MiniMax Responses API does
  not require any of these. The tests fail loudly if any of
  these fields appear in the request body.
- The library always sends ``model``, ``instructions``,
  ``input``, ``tools``, ``tool_choice: "auto"``, and
  ``reasoning: {"effort": <value>}`` for the M3 reasoning
  path.
- Auth comes from the env var (``MINIMAX_API_KEY`` primary,
  ``M3_API_KEY`` legacy) or the explicit ``api_key=``
  constructor arg. The Authorization header is
  ``Bearer <key>``.
- A malformed provider response (no function_call, an
  invalid verdict, unparseable arguments JSON) raises
  ``JudgeError`` with a clear message, not a silent
  fallback.
- LEFT, TIE, RIGHT verdicts are mapped correctly.
- The connector preserves the provider / model / reasoning
  metadata on the returned ``Judgment`` for reproducibility.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import pytest

from pairwise_rank.providers import JudgeError
from pairwise_rank.providers.base import JudgmentRequest
from pairwise_rank.providers.MiniMax import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TOOL_NAME,
    ENV_API_KEY_LEGACY,
    ENV_API_KEY_PRIMARY,
    PROVIDER_NAME,
    MiniMaxJudge,
    _verdict_tool_schema,
)


# -- Helpers ---------------------------------------------------------------

class _Recorder:
    """Mock HTTP transport. Captures the call args and returns
    a configurable response."""

    def __init__(self, response: Any = b"{}", status: int = 200):
        self.response = response
        self.status = status
        self.calls: list[dict] = []

    def __call__(self, *, url: str, body: bytes, headers, timeout: float) -> bytes:
        self.calls.append({
            "url": url,
            "body": body,
            "headers": dict(headers),
            "timeout": timeout,
        })
        if isinstance(self.response, (bytes, bytearray)):
            return bytes(self.response)
        if isinstance(self.response, dict):
            return json.dumps(self.response).encode("utf-8")
        raise TypeError(f"unsupported response type: {type(self.response)}")


def _function_call_response(verdict: str, *, reasoning_text: str = "thought") -> dict:
    """Build a minimal but realistic MiniMax Responses output."""
    return {
        "id": "resp_abc123",
        "model": DEFAULT_MODEL,
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "content": [
                    {"type": "reasoning_text", "text": reasoning_text},
                ],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "name": DEFAULT_TOOL_NAME,
                "arguments": json.dumps({"verdict": verdict}),
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


# -- Request-body contract -------------------------------------------------

@pytest.fixture
def connector() -> MiniMaxJudge:
    rec = _Recorder(response=_function_call_response("LEFT"))
    return MiniMaxJudge(api_key="test-key", http_post=rec)


def test_request_omits_temperature_top_p_and_other_sampling_knobs(connector):
    rec = connector._http_post
    request = JudgmentRequest(
        left="alpha", right="beta", instructions="compare",
    )
    connector.judge(request)
    assert len(rec.calls) == 1
    body = json.loads(rec.calls[0]["body"].decode("utf-8"))
    for forbidden in ("temperature", "top_p", "max_p", "seed", "logit_bias",
                       "frequency_penalty", "presence_penalty"):
        assert forbidden not in body, (
            f"connector must not send {forbidden!r}; "
            f"see AGENTS.md section 9 (provider philosophy)"
        )


def test_request_includes_required_fields(connector):
    rec = connector._http_post
    connector.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    body = json.loads(rec.calls[0]["body"].decode("utf-8"))
    assert body["model"] == DEFAULT_MODEL
    assert body["instructions"] == "x"
    assert isinstance(body["input"], list)
    assert body["tool_choice"] == "auto"
    assert body["reasoning"] == {"effort": DEFAULT_REASONING_EFFORT}
    # One tool, the verdict recorder.
    assert len(body["tools"]) == 1
    tool = body["tools"][0]
    assert tool["type"] == "function"
    assert tool["name"] == DEFAULT_TOOL_NAME
    assert "verdict" in tool["parameters"]["properties"]
    assert tool["parameters"]["properties"]["verdict"]["enum"] == ["LEFT", "TIE", "RIGHT"]
    assert tool["parameters"]["required"] == ["verdict"]


def test_request_user_message_references_candidate_ids(connector):
    rec = connector._http_post
    connector.judge(JudgmentRequest(left="alpha", right="beta", instructions="x"))
    body = json.loads(rec.calls[0]["body"].decode("utf-8"))
    user_msg = body["input"][0]
    assert user_msg["role"] == "user"
    text = " ".join(c["text"] for c in user_msg["content"]
                    if c.get("type") == "input_text")
    assert "alpha" in text
    assert "beta" in text


def test_request_image_parts_are_passed_through_unchanged(connector):
    img = {
        "type": "input_image",
        "image_url": {
            "url": "data:image/png;base64,XYZ",
            "detail": "high",
        },
    }
    req = JudgmentRequest(left="a", right="b", instructions="x", images=[img])
    connector.judge(req)
    body = json.loads(connector._http_post.calls[0]["body"].decode("utf-8"))
    contents = body["input"][0]["content"]
    image_parts = [c for c in contents if c.get("type") == "input_image"]
    assert len(image_parts) == 1
    assert image_parts[0] == img


def test_request_url_targets_v1_responses(connector):
    rec = connector._http_post
    connector.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert rec.calls[0]["url"].endswith("/v1/responses")


def test_request_sends_bearer_authorization(connector):
    rec = connector._http_post
    connector.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert rec.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert rec.calls[0]["headers"]["Content-Type"] == "application/json"


# -- Verdict mapping -------------------------------------------------------

@pytest.mark.parametrize("verdict", ["LEFT", "TIE", "RIGHT"])
def test_verdict_left_tie_right_map_correctly(verdict):
    rec = _Recorder(response=_function_call_response(verdict))
    c = MiniMaxJudge(api_key="k", http_post=rec)
    j = c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert j.verdict == verdict


def test_response_preserves_provider_metadata():
    rec = _Recorder(response=_function_call_response("LEFT"))
    c = MiniMaxJudge(api_key="k", http_post=rec)
    j = c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert j.provider == PROVIDER_NAME
    assert j.model == DEFAULT_MODEL
    assert j.reasoning_effort == DEFAULT_REASONING_EFFORT
    # Raw response is retained for audit.
    assert isinstance(j.raw, dict)
    assert j.raw["id"] == "resp_abc123"


def test_response_reasoning_concatenated_from_reasoning_items():
    rec = _Recorder(response=_function_call_response(
        "LEFT", reasoning_text="I thought about it. "
    ))
    c = MiniMaxJudge(api_key="k", http_post=rec)
    j = c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert j.reasoning == "I thought about it. "


def test_response_without_reasoning_item_yields_empty_reasoning():
    response = {
        "id": "resp_abc",
        "model": DEFAULT_MODEL,
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "name": DEFAULT_TOOL_NAME,
            "arguments": json.dumps({"verdict": "TIE"}),
        }],
    }
    c = MiniMaxJudge(api_key="k", http_post=_Recorder(response=response))
    j = c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert j.verdict == "TIE"
    assert j.reasoning == ""


# -- Malformed response handling ------------------------------------------

def test_missing_function_call_raises_judge_error():
    response = {
        "id": "resp_abc",
        "model": DEFAULT_MODEL,
        "output": [{"type": "message", "content": []}],
    }
    c = MiniMaxJudge(api_key="k", http_post=_Recorder(response=response))
    with pytest.raises(JudgeError) as excinfo:
        c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert "function_call" in str(excinfo.value)


def test_missing_output_list_raises_judge_error():
    response = {"id": "resp", "model": DEFAULT_MODEL}  # no 'output' key
    c = MiniMaxJudge(api_key="k", http_post=_Recorder(response=response))
    with pytest.raises(JudgeError) as excinfo:
        c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert "output" in str(excinfo.value)


def test_invalid_verdict_in_tool_call_raises_judge_error():
    response = _function_call_response("MAYBE")
    c = MiniMaxJudge(api_key="k", http_post=_Recorder(response=response))
    with pytest.raises(JudgeError) as excinfo:
        c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert "MAYBE" in str(excinfo.value) or "verdict" in str(excinfo.value)


def test_unparseable_arguments_json_raises_judge_error():
    response = _function_call_response("LEFT")
    response["output"][1]["arguments"] = "{not json"
    c = MiniMaxJudge(api_key="k", http_post=_Recorder(response=response))
    with pytest.raises(JudgeError):
        c.judge(JudgmentRequest(left="a", right="b", instructions="x"))


def test_unparseable_response_body_raises_judge_error():
    c = MiniMaxJudge(api_key="k", http_post=_Recorder(response=b"not json at all"))
    with pytest.raises(JudgeError):
        c.judge(JudgmentRequest(left="a", right="b", instructions="x"))


# -- Auth ------------------------------------------------------------------

def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY_PRIMARY, "from-env-primary")
    monkeypatch.setenv(ENV_API_KEY_LEGACY, "from-env-legacy")
    rec = _Recorder(response=_function_call_response("LEFT"))
    c = MiniMaxJudge(api_key="explicit-key", http_post=rec)
    c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert rec.calls[0]["headers"]["Authorization"] == "Bearer explicit-key"


def test_primary_env_var_used_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY_PRIMARY, "primary-key")
    monkeypatch.delenv(ENV_API_KEY_LEGACY, raising=False)
    rec = _Recorder(response=_function_call_response("LEFT"))
    c = MiniMaxJudge(http_post=rec)
    c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert rec.calls[0]["headers"]["Authorization"] == "Bearer primary-key"


def test_legacy_env_var_used_as_fallback(monkeypatch):
    monkeypatch.delenv(ENV_API_KEY_PRIMARY, raising=False)
    monkeypatch.setenv(ENV_API_KEY_LEGACY, "legacy-key")
    rec = _Recorder(response=_function_call_response("LEFT"))
    c = MiniMaxJudge(http_post=rec)
    c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    assert rec.calls[0]["headers"]["Authorization"] == "Bearer legacy-key"


def test_missing_api_key_raises_judge_error(monkeypatch):
    monkeypatch.delenv(ENV_API_KEY_PRIMARY, raising=False)
    monkeypatch.delenv(ENV_API_KEY_LEGACY, raising=False)
    with pytest.raises(JudgeError) as excinfo:
        MiniMaxJudge()
    assert ENV_API_KEY_PRIMARY in str(excinfo.value)


# -- Configuration validation ---------------------------------------------

def test_invalid_reasoning_effort_rejected():
    with pytest.raises(ValueError):
        MiniMaxJudge(api_key="k", reasoning_effort="ultra")


def test_tool_schema_is_pinned_to_recording_verb():
    schema = _verdict_tool_schema(DEFAULT_TOOL_NAME)
    assert schema["type"] == "function"
    assert schema["name"] == DEFAULT_TOOL_NAME
    assert "Record" in schema["description"]
    # Verdict is the only parameter.
    assert list(schema["parameters"]["properties"].keys()) == ["verdict"]
    assert schema["parameters"]["required"] == ["verdict"]
    # TIE is a first-class option.
    assert "TIE" in schema["parameters"]["properties"]["verdict"]["enum"]


def test_custom_tool_name_appears_in_request():
    rec = _Recorder(response=_function_call_response("LEFT"))
    c = MiniMaxJudge(api_key="k", tool_name="my_custom_tool", http_post=rec)
    c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    body = json.loads(rec.calls[0]["body"].decode("utf-8"))
    assert body["tools"][0]["name"] == "my_custom_tool"


def test_custom_model_appears_in_request():
    rec = _Recorder(response=_function_call_response("LEFT"))
    c = MiniMaxJudge(api_key="k", model="MiniMax-M3-preview", http_post=rec)
    c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    body = json.loads(rec.calls[0]["body"].decode("utf-8"))
    assert body["model"] == "MiniMax-M3-preview"


def test_reasoning_effort_none_is_accepted():
    rec = _Recorder(response=_function_call_response("LEFT"))
    c = MiniMaxJudge(api_key="k", reasoning_effort="none", http_post=rec)
    c.judge(JudgmentRequest(left="a", right="b", instructions="x"))
    body = json.loads(rec.calls[0]["body"].decode("utf-8"))
    assert body["reasoning"] == {"effort": "none"}
