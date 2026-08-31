"""Connector base types: JudgmentRequest, Judgment, JudgeConnector.

This module is the entire provider abstraction. It defines
the request / response shape and the connector Protocol.

The connector abstraction is intentionally narrow:

    judge(request) -> judgment

A provider implementation supplies the HTTP transport, the
structured-output schema, and any provider-specific
configuration. The canonical `Judgment` shape is fixed by
this library. Provider-specific quirks (e.g. tool-call
response shape, multimodal message format, reasoning field
naming) are adapted inside the provider implementation, not
by mutating the canonical contract.

Do not add:

- a registry
- a factory
- capability negotiation
- a base class for providers
- a global configuration object
- a session / connection pool
- retry logic with exponential backoff (the protocol layer
  and the caller decide how to retry)

A future agent can read this file in one screen and add
another provider without consulting anything else in the
package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence


class JudgeError(Exception):
    """Raised by a JudgeConnector when a judgment cannot be
    obtained. Subclasses or instances may carry provider /
    model / response context for debugging.

    Library callers should treat any JudgeError as a failed
    judgment, not as a verdict. The protocol layer does not
    catch this; it propagates so the caller can decide whether
    to retry, log, or fail the run.
    """


@dataclass(frozen=True)
class JudgmentRequest:
    """All the fields needed to make a single judgment call.

    The connector is responsible for translating these fields
    into the provider's request shape. The fields here are
    only what the protocol genuinely needs; provider-specific
    configuration (model identifier, reasoning effort, base
    URL, timeouts) lives on the connector instance, not on
    every request.

    left, right       : the displayed candidate ids. The
                        protocol does not prescribe how the
                        connector presents them to the model
                        (text-only, image attachments, etc.).
                        For multimodal, see `images`.
    instructions      : the user-supplied system-style
                        instructions that define the construct
                        being judged. The connector passes this
                        through verbatim. The protocol layer
                        never inspects or modifies it.
    images            : optional multimodal content. Each
                        element is a provider-ready dict (e.g.
                        MiniMax `input_image` shape). The
                        connector is responsible for placing
                        these into the provider's content list.
                        Empty list means text-only.
    """
    left: str
    right: str
    instructions: str
    images: Sequence[Mapping[str, object]] = field(default_factory=tuple)


@dataclass(frozen=True)
class Judgment:
    """The result of a single judgment call.

    Fields:
      verdict          : one of "LEFT", "TIE", "RIGHT". Empty
                         string is reserved for "not yet
                         judged" but a real connector must
                         always populate this with a real
                         verdict or raise.
      reasoning        : free-form audit text (the model's
                         reasoning content, if the provider
                         exposes it). Empty string if the
                         provider does not expose reasoning.
                         Stored on the Observation as audit
                         metadata; never enters the ranking
                         model. See protocol.Observation.
      provider         : provider identifier ("MiniMax", ...).
                         Recorded for reproducibility.
      model            : model identifier actually used for
                         the call. Recorded for reproducibility.
      reasoning_effort : the reasoning-control value sent (e.g.
                         "high" for M3 Adaptive Thinking), or
                         empty string if the provider has no
                         reasoning control.
      raw              : optional raw provider response,
                         retained for debugging and audit. Do
                         not log it by default; it may contain
                         user-provided content.
    """
    verdict: str
    reasoning: str = ""
    provider: str = ""
    model: str = ""
    reasoning_effort: str = ""
    raw: Optional[Mapping[str, object]] = None


class JudgeConnector(Protocol):
    """The connector contract.

    A provider implements one operation: `judge(request) ->
    Judgment`. The connector does not decide which pair to
    compare, when to stop, or what to do with the result;
    those are the protocol's and the caller's responsibility.

    Connectors are not required to be threadsafe. A typical
    workflow constructs one connector and uses it across a
    sequential tournament. Concurrent use is the caller's
    problem.
    """
    def judge(self, request: JudgmentRequest) -> Judgment: ...
