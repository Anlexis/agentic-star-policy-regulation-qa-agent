"""AgentCore Platform v1.0"""

# GOV-C2-001 — GenerateAnswerNode
# Inner domain node 4 (LLM, grounded): generate a policy answer grounded ONLY
# in the filtered passages, with citations.
#
# Implementation: deterministic, extractive grounded synthesis over the
# filtered passages (the passages that cleared the score_threshold). Production
# wires the real LLM here via the constructor-injected config
# (self._config["configurable"]["system_prompt"], rendered from
# prompts/gov_qa.j2) at temperature 0.0 — the installed framework SDK ships
# no LLM client.
#
# Grounding discipline (life/safety-adjacent domain): when NO passage cleared
# the bar (grounded is false / filtered set empty) the node DECLINES — it never
# fabricates a policy answer.
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

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

_DECLINE_MESSAGE = (
    "The current government policy knowledge base does not contain a passage "
    "that confidently answers this question. To avoid providing incorrect "
    "regulatory guidance, no answer is given. Please contact the responsible "
    "administrative office for an authoritative response."
)


def _synthesise_grounded_answer(query: str, passages: List[Dict[str, Any]]) -> str:
    """Build a deterministic answer grounded strictly in the filtered passages.

    Production replaces this with the LLM call (prompts/gov_qa.j2, temp 0.0).
    """
    citation_ids = ", ".join(p.get("id", "?") for p in passages)
    lines = [
        f"Question: {query}",
        "",
        ("Based strictly on the retrieved government policy sources, the " "following applies:"),
        "",
    ]
    for p in passages:
        lines.append(f"- [{p.get('id', '?')}] {p.get('source', 'unknown source')}:")
        lines.append(f"  {p.get('text', '').strip()}")
    lines += [
        "",
        f"(出典 / Sources: {citation_ids})",
        (
            "This response is drawn only from the cited policy passages. For an "
            "authoritative determination on a specific case, contact the "
            "responsible administrative office."
        ),
    ]
    return "\n".join(lines)


class GenerateAnswerNode(FunctionNode):
    """Generate a grounded, cited policy answer (or decline).

    Deterministic extractive synthesis over the filtered passages.
    Production: the constructor-injected config
    (self._config["configurable"].get("system_prompt", "")) drives an LLM
    at temperature 0.0.

    Inner node — ANONYMOUS trust (see module docstring).

    Configuration is injected through the constructor:
    DomainWorkflowGraph.register_nodes() passes the graph's config dict (the
    ``{"configurable": {...}}`` shape built by
    PolicyQAGraphNode._parent_config() from config/config.yaml).

    Input state keys:
        filtered_docs: str   — JSON-serialised passages clearing the bar
        query:         str   — validated policy question

    Output state keys (partial dict):
        answer:    str   — grounded answer or the decline message
        citations: str   — JSON-serialised [{id, source}]
        status:    str
        error_log: list[str]  (only on ERROR)
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

        emit_progress("Composing the answer...")
        query = state.get("query") or state.get("validated_input") or ""
        passages: List[Dict[str, Any]] = from_json(state.get("filtered_docs"), []) or []

        # Read prompt + temperature from the constructor-injected config
        # (production LLM wiring point).
        configurable = (self._config or {}).get("configurable", {})
        _system_prompt = configurable.get("system_prompt", "")  # noqa: F841
        _temperature = configurable.get("temperature", 0.0)  # noqa: F841

        # ── Grounding discipline: decline when nothing cleared the bar ─────────
        if not passages:
            logger.info("GenerateAnswerNode: no grounded passages — declining")
            emit_trace_event(
                "generate_answer_declined",
                {"reason": "no_grounded_passages", "grounded": False},
                state,
            )
            return {
                "answer": _DECLINE_MESSAGE,
                "citations": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        # ── Grounded synthesis ────────────────────────────────────────────────
        answer = _synthesise_grounded_answer(query, passages)
        citations = [{"id": p.get("id", "?"), "source": p.get("source", "unknown source")} for p in passages]

        logger.info(
            "GenerateAnswerNode: grounded answer generated from %d passage(s)",
            len(passages),
        )
        emit_trace_event(
            "generate_answer_complete",
            {
                "grounded": True,
                "passage_count": len(passages),
                "citation_ids": [c["id"] for c in citations],
            },
            state,
        )

        return {
            "answer": answer,
            "citations": to_json(citations),
            "status": AgentStatus.SUCCESS.value,
        }
