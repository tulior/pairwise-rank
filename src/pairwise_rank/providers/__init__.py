"""Provider connectors for the judgment interface.

This package is a thin, opt-in layer between the protocol
(`pairwise_rank.protocol.JudgeFn`) and the external LLM API.
It is not a provider framework. There is one Protocol,
`JudgeConnector`, and one shipped implementation, `MiniMaxJudge`.

Why this module exists
======================

The protocol's `JudgeFn` is `Callable[[str, str], JudgeReturn]`,
which is exactly what the protocol layer needs: a function that
turns a pair of candidate ids into a verdict. Real LLM calls
need more than `(left, right)`: a system instructions string,
optional multimodal inputs, a structured-output schema, and
reasoning configuration. The provider connector is the place
where those extra fields are configured, sent to the API, and
parsed back into the protocol's canonical `Judgment` shape.

The connector does not own which pair to compare, how many
repeats to run, when to stop, or how the rankings are computed.
Those belong to the caller and to the protocol / statistical
layer, not here.

Adding another provider
=======================

To add another provider, see AGENTS.md section 10
("Adding a provider"). The shape is:

    src/pairwise_rank/providers/<name>.py

    from .base import JudgmentRequest, Judgment, JudgeError

    class <Name>Judge:
        def __init__(self, ...): ...
        def judge(self, request: JudgmentRequest) -> Judgment: ...

Add a contract test under `tests/test_providers/test_<name>.py`
that mocks the HTTP layer. Do not require live API access for
the default suite.
"""
from .base import (
    Judgment,
    JudgmentRequest,
    JudgeConnector,
    JudgeError,
)

__all__ = [
    "Judgment",
    "JudgmentRequest",
    "JudgeConnector",
    "JudgeError",
]
