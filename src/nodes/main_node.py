"""AgentCore Platform v1.0"""

# GOV-C2-001 — DEPRECATED: MainNode
#
# This file is superseded by the Cat 2 nested architecture.  The `main`
# slot of PolicyRegulationQAAgent is now filled by PolicyQAGraphNode
# (src/graph/graph.py), which delegates the RAG domain workflow to
# DomainWorkflowGraph (domain nodes in src/nodes/).
#
# This stub remains to avoid import errors in any code that may reference
# this module path.  It is NOT imported by graph.py or any other template code.
#
# DO NOT USE -- retained only while the sample files reference this path.

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

_DEPRECATION_MSG = (
    "DEPRECATED: MainNode is superseded by the Cat 2 DomainWorkflowGraph pipeline "
    "(PolicyQAGraphNode -> DomainWorkflowGraph). "
    "This stub is retained for import compatibility only."
)


class MainNode(FunctionNode):
    """DEPRECATED -- superseded by Cat 2 domain nodes in DomainWorkflowGraph."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # Audit event required in every execute().
        # The request was already found unacceptable upstream: this run
        # completes without a result, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_trace_event(
            "main_node_deprecated_called",
            {
                "warning": "MainNode is deprecated and not part of the Cat 2 pipeline",
                "replacement": "PolicyQAGraphNode + DomainWorkflowGraph",
            },
            state,
        )
        return {
            "status": AgentStatus.SUCCESS.value,
            "result": _DEPRECATION_MSG,
        }
