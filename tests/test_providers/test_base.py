"""Tests for the provider base types: JudgmentRequest, Judgment, JudgeError.

These tests do not require network access. They pin the
canonical shape that providers must map into.
"""
from __future__ import annotations

import pytest

from pairwise_rank.providers.base import (
    JudgeError,
    Judgment,
    JudgmentRequest,
)


def test_judgment_request_minimal():
    req = JudgmentRequest(left="alpha", right="beta", instructions="compare")
    assert req.left == "alpha"
    assert req.right == "beta"
    assert req.instructions == "compare"
    assert req.images == ()


def test_judgment_request_with_images():
    img = {"type": "input_image", "image_url": {"url": "data:image/png;base64,..."}}
    req = JudgmentRequest(left="a", right="b", instructions="x", images=[img])
    assert len(req.images) == 1
    assert req.images[0]["type"] == "input_image"


def test_judgment_request_is_frozen():
    req = JudgmentRequest(left="a", right="b", instructions="x")
    with pytest.raises(Exception):
        req.left = "z"  # type: ignore[misc]


def test_judgment_minimal_verdict_only():
    j = Judgment(verdict="LEFT")
    assert j.verdict == "LEFT"
    assert j.reasoning == ""
    assert j.provider == ""
    assert j.model == ""
    assert j.reasoning_effort == ""
    assert j.raw is None


def test_judgment_full():
    j = Judgment(
        verdict="TIE",
        reasoning="both look similar",
        provider="MiniMax",
        model="MiniMax-M3",
        reasoning_effort="high",
        raw={"x": 1},
    )
    assert j.verdict == "TIE"
    assert j.reasoning == "both look similar"
    assert j.provider == "MiniMax"
    assert j.model == "MiniMax-M3"
    assert j.reasoning_effort == "high"
    assert j.raw == {"x": 1}


def test_judge_error_is_exception():
    assert issubclass(JudgeError, Exception)
    with pytest.raises(JudgeError) as excinfo:
        raise JudgeError("test")
    assert "test" in str(excinfo.value)
