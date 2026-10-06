"""AgentCore Platform v1.0"""

# GOV-C2-001 — OutputFormatNode (inner domain node 5, last in DomainWorkflowGraph)
# Assembles the final, cited policy answer document from `answer` and
# `citations`.  This is the last inner node — it produces the answer_document
# string that the outer PostProcessNode's output gate will scan.
#
# Inner node — ANONYMOUS trust (trust is enforced at the outer pre_process).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress

from src.schemas.state import from_json

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72


def _assemble_document(
    query: str,
    answer: str,
    citations: List[Dict[str, Any]],
    grounded: bool,
) -> str:
    """Assemble the final cited policy answer document."""
    lines = [
        _SEPARATOR,
        "POLICY & REGULATION Q&A",
        _SEPARATOR,
        "",
        "Question:",
        f"  {query or '(question unavailable)'}",
        "",
        "Answer:",
        answer.strip() or "(no answer content)",
        "",
        _SUBSEP,
        "Sources / 出典",
        _SUBSEP,
    ]
    if citations:
        for c in citations:
            lines.append(f"  [{c.get('id', '?')}] {c.get('source', 'unknown source')}")
    else:
        lines.append("  (none — answer not grounded in a KB passage)")
    lines += [
        _SUBSEP,
        f"Grounded: {'YES' if grounded else 'NO'}",
        _SEPARATOR,
    ]
    return "\n".join(lines)


class OutputFormatNode(FunctionNode):
    """Assemble the final cited policy answer document (inner domain node).

    Reads answer and citations from State, renders the full cited answer
    document, and writes it to answer_document (and result) for the outer
    PostProcessNode.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        answer:    str   — grounded answer or decline message
        citations: str   — JSON-serialised [{id, source}]
        query:     str   — validated policy question
        grounded:  bool  — whether the answer was grounded

    Output state keys (partial dict):
        answer_document: str
        result:          str  (same as answer_document — backbone convention)
        status:          str
        error_log:       list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without a result, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Formatting the response...")
        query = state.get("query") or state.get("validated_input") or ""
        answer = state.get("answer") or ""
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []
        grounded = bool(state.get("grounded"))

        if not answer.strip():
            logger.warning("OutputFormatNode: answer is empty — using fallback")
            answer = "No answer content was produced. Please retry or contact the " "responsible administrative office."

        document = _assemble_document(query, answer, citations, grounded)

        logger.info(
            "OutputFormatNode: document assembled (%d chars, %d citation(s), grounded=%s)",
            len(document),
            len(citations),
            grounded,
        )
        emit_trace_event(
            "output_format_complete",
            {
                "document_length": len(document),
                "citation_count": len(citations),
                "grounded": grounded,
            },
            state,
        )

        return {
            "answer_document": document,
            "result": document,
            "status": AgentStatus.SUCCESS.value,
        }
