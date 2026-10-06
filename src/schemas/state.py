"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# GOV-C2-001 — Policy & Regulation Q&A Agent (Cat 2 RAG)
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# Serialisation contract: all dict/list-valued fields are stored as JSON-
# serialized Optional[str].  Use to_json() / from_json() helpers below
# at every producer and consumer node — one contract end-to-end.
# Never type a dict/list field as a bare dict/list; that causes msgpack
# serialization failures.

import json
import math
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse a caller-controlled numeric: FINITE number within [lo, hi], else None.

    Rejects bools, strings, non-numerics, and — critically — non-finite values:
    Python's json module happily parses bare NaN/Infinity in request bodies, and
    IEEE NaN comparisons are always False, which would turn a threshold check
    into a silent fail-open. Every caller-supplied number must come through here
    (or an equivalent explicit finite check) and fail CLOSED on violation.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for GOV-C2-001 Policy & Regulation Q&A Agent.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.

    Serialisation contract: dict/list fields use JSON-serialized Optional[str].
    formatted_output is NOT re-declared here — it is inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode (pre_process backbone)
    # ------------------------------------------------------------------

    # Validated and normalised policy question string.
    # Produced by PreProcessNode; consumed by inner InputValidateNode.
    validated_input: Optional[str]

    # JSON-serialised channel/request metadata dict (stored as str).
    # Shape: {"source": str, "channel": str}
    enriched_context: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # Cleaned/normalised policy question, produced by InputValidateNode.
    query: Optional[str]

    # JSON-serialised list of VALIDATED caller-supplied policy passages
    # (stored as str). Present only when the caller provided
    # input_context["policy_passages"] and every entry passed the ingest
    # validation contract in InputValidateNode. When present, retrieval runs
    # over this corpus instead of the built-in sample corpus.
    caller_corpus: Optional[str]

    # JSON-serialised dict of VALIDATED caller retrieval overrides
    # (stored as str). Shape: {"top_k": int, "score_threshold": float} —
    # both optional, both bounds-checked at ingest (the threshold may only
    # tighten the configured grounding bar, never loosen it).
    retrieval_overrides: Optional[str]

    # JSON-serialised list of candidate KB passages (stored as str).
    # Each item: {"id": str, "source": str, "text": str, "score": float}
    # Produced by RetrieveNode (top_k candidates).
    retrieved_docs: Optional[str]

    # Number of candidate passages returned by RetrieveNode.
    retrieved_count: Optional[int]

    # JSON-serialised list of passages that cleared the score_threshold
    # after reranking (stored as str).  Produced by RerankFilterNode.
    filtered_docs: Optional[str]

    # True iff at least one passage cleared score_threshold (grounding present).
    grounded: Optional[bool]

    # Grounded answer text produced by GenerateAnswerNode (or a decline
    # message when no passage cleared the threshold).
    answer: Optional[str]

    # JSON-serialised list of source citations (stored as str).
    # Each item: {"id": str, "source": str}.  Produced by GenerateAnswerNode.
    citations: Optional[str]

    # Final formatted, cited answer document assembled by OutputFormatNode.
    answer_document: Optional[str]

    # ------------------------------------------------------------------
    # Outer layer — set by PostProcessNode (post_process backbone, output gate)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller.
    # Set to the gate-checked answer_document by PostProcessNode.
    # formatted_output (from AgentState) is also set by PostProcessNode.
    result: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]

    # Set when the run completes WITHOUT carrying out the request because the
    # caller's input could not be accepted as written - a rejection the caller
    # can correct and retry. The run still completes: nothing is processed, no
    # product is assembled, and the domain audit event for the rejection is
    # still emitted. Carrying this as a completion marker rather than a terminal
    # error is what lets the caller see the reason and send a corrected request
    # on the same conversation.
    #
    # Content the agent refuses outright, and a breach of a contract the caller
    # cannot influence, are NOT reported here - those stay terminal.
    error_code: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState
