"""AgentCore Platform v1.0"""

# GOV-C2-001 — RerankFilterNode
# Inner domain node 3: rerank retrieved passages and drop everything below the
# retrieval score_threshold.
#
# Government guidance is life/safety-adjacent, so the grounding bar is raised
# (score_threshold >= 0.75; config default 0.78). Only high-confidence policy
# passages are allowed to ground an answer; when nothing clears the bar the
# filtered set is empty and the downstream GenerateAnswerNode declines rather
# than guessing.
#
# Inner node — ANONYMOUS trust (trust is enforced at the outer pre_process).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress

from src.schemas.state import finite_in_range, from_json, to_json

logger = logging.getLogger(__name__)

# Default grounding bar — overridden by config retrieval.score_threshold.
# Life/safety-adjacent domain → never below 0.75.
_DEFAULT_SCORE_THRESHOLD = 0.78
_MIN_ALLOWED_THRESHOLD = 0.75


class RerankFilterNode(FunctionNode):
    """Rerank candidate passages and filter by score_threshold.

    Inner node — ANONYMOUS trust (see module docstring).

    Configuration is injected through the constructor:
    DomainWorkflowGraph.register_nodes() passes the graph's config dict (the
    ``{"configurable": {...}}`` shape built by
    PolicyQAGraphNode._parent_config() from config/config.yaml). A validated
    caller ``retrieval.score_threshold`` override (state["retrieval_overrides"])
    may only TIGHTEN the bar: the effective threshold is the maximum of the
    configured value, the caller override, and the hard floor.

    Input state keys:
        retrieved_docs:      str — JSON-serialised candidate list
        retrieval_overrides: str — JSON dict of validated overrides (optional)

    Output state keys (partial dict):
        filtered_docs: str   — JSON-serialised passages clearing the bar
        grounded:      bool  — True iff at least one passage cleared the bar
        status:        str
        error_log:     list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Store the graph-level config dict carrying the ``configurable`` key."""
        super().__init__()
        self._config = config

    def execute(self, state: AgentState) -> dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without a result, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Ranking the results...")
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_docs"), []) or []

        configurable = (self._config or {}).get("configurable", {})
        raw_threshold = configurable.get("score_threshold", _DEFAULT_SCORE_THRESHOLD)
        # Defense in depth: the configured value must itself be a finite number
        # in [0, 1] — anything else falls back to the declared default.
        parsed_threshold = finite_in_range(raw_threshold, 0.0, 1.0)
        threshold = parsed_threshold if parsed_threshold is not None else _DEFAULT_SCORE_THRESHOLD

        # A validated caller override may only TIGHTEN the bar (max() below);
        # re-parse defensively — a non-finite value can never loosen the gate.
        overrides = from_json(state.get("retrieval_overrides"), {}) or {}
        if "score_threshold" in overrides:
            caller_threshold = finite_in_range(overrides["score_threshold"], 0.0, 1.0)
            if caller_threshold is not None:
                threshold = max(threshold, caller_threshold)

        # Enforce the life/safety-adjacent floor: never let config drop the bar
        # below 0.75 for this template.
        if threshold < _MIN_ALLOWED_THRESHOLD:
            threshold = _MIN_ALLOWED_THRESHOLD

        # Rerank by score desc (stable, deterministic by id) then apply the bar.
        ranked = sorted(candidates, key=lambda d: (-float(d.get("score", 0.0)), d.get("id", "")))
        filtered: List[Dict[str, Any]] = [d for d in ranked if float(d.get("score", 0.0)) >= threshold]
        grounded = len(filtered) > 0

        logger.info(
            "RerankFilterNode: candidates=%d filtered=%d threshold=%.2f grounded=%s",
            len(candidates),
            len(filtered),
            threshold,
            grounded,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "candidate_count": len(candidates),
                "filtered_count": len(filtered),
                "score_threshold": threshold,
                "grounded": grounded,
            },
            state,
        )

        return {
            "filtered_docs": to_json(filtered),
            "grounded": grounded,
            "status": AgentStatus.SUCCESS.value,
        }
