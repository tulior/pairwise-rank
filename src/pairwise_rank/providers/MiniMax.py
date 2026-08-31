"""MiniMax judge connector.

Thin, stdlib-only implementation of `JudgeConnector` for the
MiniMax Responses API. One file, one class.

Reference (verify before changing model identifier or
reasoning field):
    https://platform.minimax.io/docs/api-reference/responses-create

Decisions recorded here
=======================

1. Endpoint: ``POST https://api.minimax.io/v1/responses`` (the
   Responses API, OpenAI Responses compatible). The OpenAI
   Chat Completions endpoint also exists but is a different
   shape; we use Responses for the structured-output
   tool-call flow.

2. Model: ``MiniMax-M3`` is the current reasoning-capable
   model. The list of supported models changes; before
   pinning a different model identifier, re-check the docs
   above. Do not scatter model identifiers across the
   codebase.

3. Reasoning: the field is ``reasoning: {"effort": "<v>"}``.
   For ``MiniMax-M3`` the values ``minimal``, ``low``,
   ``medium``, and ``high`` all enable Adaptive Thinking but
   do not tune its depth. The default is ``none`` (reasoning
   disabled). We set ``effort="high"`` to enable reasoning
   at the maximum compatibility setting the API exposes. The
   library deliberately does not interpret ``high`` as a
   depth guarantee.

4. Sampling defaults: the library does **not** send
   ``temperature``, ``top_p``, ``max_p``, ``seed``,
   ``penalties``, or any other sampling knob. The provider
   default applies. This is intentional and recorded in
   AGENTS.md section 9. If a future provider requires one
   of these fields, document the requirement in the
   connector and explain why.

5. Structured output: one tool, named
   ``record_posterior_comparison``, with a single ``verdict``
   parameter of enum ``{LEFT, TIE, RIGHT}``. The tool name
   and description are part of the rubric: passive recording
   verb, the estimand stated in the description, and TIE as
   a first-class option. ``tool_choice: "auto"`` is sent
   because the documented values are ``"none"`` and
   ``"auto"``; we do not invent a named-function-forcing
   object.

6. Multimodal: image parts use ``input_image`` with an
   object-valued ``image_url`` of the form
   ``{"url": "data:image/png;base64,...", "detail": "high"}``.
   The ``detail: "high"`` is required for fine-grained
   inputs.

7. Authentication: read from environment. The primary env
   var is ``MINIMAX_API_KEY``. A legacy alias ``M3_API_KEY``
   is also accepted. The connector never reads credentials
   from any other source and never writes them to disk or
   logs.

8. No retry / no backoff inside the connector. Retries are
   the caller's job. The library deliberately keeps the
   connector small.

9. Error policy: a malformed response (no function_call, an
   invalid verdict, an unreadable argument JSON) raises
   ``JudgeError`` with a one-line context message. HTTP
   errors are propagated as ``urllib.error.HTTPError`` with
   the original response body. Do not silently substitute a
   default verdict; that would corrupt the data.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Optional, Sequence

from .base import JudgeConnector, JudgeError, Judgment, JudgmentRequest


# -- Configuration ---------------------------------------------------------

PROVIDER_NAME = "MiniMax"
DEFAULT_BASE_URL = "https://api.minimax.io"
DEFAULT_ENDPOINT = "/v1/responses"
DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_TOOL_NAME = "record_posterior_comparison"
DEFAULT_TIMEOUT = 60.0

# Env vars accepted, in priority order. The first one with a
# non-empty value wins. Documented in the module docstring.
ENV_API_KEY_PRIMARY = "MINIMAX_API_KEY"
ENV_API_KEY_LEGACY = "M3_API_KEY"

# Valid verdicts. Must match pairwise_rank.protocol.VERDICT_LEVELS
# (the canonical 3-level scale).
VALID_VERDICTS = ("LEFT", "TIE", "RIGHT")


# -- Tool schema -----------------------------------------------------------

def _verdict_tool_schema(tool_name: str) -> dict:
    """The single verdict-recording tool.

    The tool name and description are part of the rubric; see
    the module docstring. The connector exposes ``tool_name``
    so tests can pin the name without changing the production
    default. The shape returned here is the documented
    MiniMax Responses-API tool object.
    """
    return {
        "type": "function",
        "name": tool_name,
        "description": (
            "Record the outcome of the pairwise comparison "
            "under the stated construct. Returns the verdict, "
            "not a re-derivation of the analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": list(VALID_VERDICTS),
                    "description": (
                        "Which side is preferred under the "
                        "construct, or TIE if there is no "
                        "material preference."
                    ),
                },
            },
            "required": ["verdict"],
            "additionalProperties": False,
        },
        "strict": True,
    }


# -- HTTP layer ------------------------------------------------------------

# Default HTTP transport. Tests can override by passing
# http_post=... to the connector.
def _default_http_post(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> bytes:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# -- Connector -------------------------------------------------------------

class MiniMaxJudge:
    """MiniMax Responses-API connector.

    Construct once, call ``judge(...)`` per pairwise row. See
    the module docstring for the full decision record.

    Auth: read from ``$MINIMAX_API_KEY`` (or ``$M3_API_KEY``)
    by default. Pass ``api_key=...`` explicitly to override
    the env var (useful in tests).

    Sampling: provider default. No temperature, no top_p.
    Reasoning: explicit ``effort="high"`` to enable Adaptive
    Thinking for M3.
    """
    provider: str = PROVIDER_NAME

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        base_url: str = DEFAULT_BASE_URL,
        endpoint: str = DEFAULT_ENDPOINT,
        tool_name: str = DEFAULT_TOOL_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        http_post: Optional[Callable[..., bytes]] = None,
    ) -> None:
        if reasoning_effort not in ("none", "minimal", "low", "medium", "high"):
            # The compatibility set is the documented enum. The
            # default is "high" so reasoning is on. Pass "none"
            # explicitly to disable.
            raise ValueError(
                f"unsupported reasoning_effort {reasoning_effort!r}; "
                f"must be one of none, minimal, low, medium, high"
            )
        self.api_key = api_key if api_key else self._read_api_key()
        if not self.api_key:
            raise JudgeError(
                f"no API key: set ${ENV_API_KEY_PRIMARY} or "
                f"${ENV_API_KEY_LEGACY} or pass api_key=..."
            )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
        self.tool_name = tool_name
        self.timeout = timeout
        self._http_post = http_post if http_post is not None else _default_http_post

    @staticmethod
    def _read_api_key() -> Optional[str]:
        for name in (ENV_API_KEY_PRIMARY, ENV_API_KEY_LEGACY):
            v = os.environ.get(name)
            if v:
                return v
        return None

    # ---- public ----

    def judge(self, request: JudgmentRequest) -> Judgment:
        body = self._build_body(request)
        raw_bytes = self._post(body)
        try:
            response = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise JudgeError(f"unparseable MiniMax response: {e}") from e
        return self._parse(response)

    # ---- internals ----

    def _build_body(self, request: JudgmentRequest) -> dict:
        # Sampling defaults are intentionally omitted. The
        # library never sends temperature / top_p / max_p /
        # seed / penalties unless the provider API requires
        # one of them. Reasoning is explicit so Adaptive
        # Thinking is enabled for M3.
        user_content: list[dict] = []
        for img in request.images:
            user_content.append(dict(img))
        user_content.append({
            "type": "input_text",
            "text": (
                f"Candidate A: {request.left}\n"
                f"Candidate B: {request.right}\n"
                "Compare them under the construct stated in the "
                "instructions and record your verdict."
            ),
        })
        return {
            "model": self.model,
            "instructions": request.instructions,
            "input": [{"role": "user", "content": user_content}],
            "tools": [_verdict_tool_schema(self.tool_name)],
            "tool_choice": "auto",
            "reasoning": {"effort": self.reasoning_effort},
        }

    def _post(self, body: dict) -> bytes:
        url = self.base_url + self.endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        try:
            return self._http_post(
                url=url,
                body=json.dumps(body).encode("utf-8"),
                headers=headers,
                timeout=self.timeout,
            )
        except urllib.error.HTTPError as e:
            # Read the body so the caller can see the provider's
            # error context. urllib closes the response on
            # context exit; read it eagerly.
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            raise JudgeError(
                f"MiniMax HTTP {e.code} {e.reason}: {err_body[:500]}"
            ) from e

    def _parse(self, response: Any) -> Judgment:
        if not isinstance(response, dict):
            raise JudgeError(
                f"MiniMax response is not a JSON object: {type(response).__name__}"
            )
        output = response.get("output")
        if not isinstance(output, list):
            raise JudgeError(
                "MiniMax response missing 'output' list; "
                f"keys present: {list(response.keys())}"
            )
        tool_call, reasoning_text = self._extract_tool_call_and_reasoning(output)
        if tool_call is None:
            types = [o.get("type") for o in output if isinstance(o, dict)]
            raise JudgeError(
                f"MiniMax response has no function_call; "
                f"output item types: {types}"
            )
        arguments = tool_call.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                raise JudgeError(
                    f"unparseable tool_call arguments: {e}; "
                    f"raw: {arguments[:200]}"
                ) from e
        if not isinstance(arguments, dict):
            raise JudgeError(
                f"tool_call arguments is not a dict: {type(arguments).__name__}"
            )
        verdict = str(arguments.get("verdict", "")).upper()
        if verdict not in VALID_VERDICTS:
            raise JudgeError(
                f"invalid verdict in tool_call: {verdict!r}; "
                f"valid: {list(VALID_VERDICTS)}"
            )
        return Judgment(
            verdict=verdict,
            reasoning=reasoning_text,
            provider=self.provider,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            raw=response,
        )

    @staticmethod
    def _extract_tool_call_and_reasoning(output: Sequence[Any]) -> tuple[Optional[dict], str]:
        """Walk the output list, return (function_call, reasoning_text).

        The MiniMax Responses-API output is an ordered list of
        items of various types. We look for the first
        ``function_call`` and concatenate any ``reasoning``
        item's text content into a single string for audit
        metadata. Items we do not recognise are ignored.
        """
        tool_call: Optional[dict] = None
        reasoning_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "function_call" and tool_call is None:
                tool_call = item
            elif t == "reasoning":
                content = item.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "reasoning_text":
                            text = part.get("text", "")
                            if isinstance(text, str):
                                reasoning_parts.append(text)
        return tool_call, "".join(reasoning_parts)
