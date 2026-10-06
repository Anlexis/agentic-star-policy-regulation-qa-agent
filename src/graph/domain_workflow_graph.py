"""AgentCore Platform v1.0"""

# GOV-C2-001 — DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the policy Q&A RAG pipeline:
#
#   START
#     → input_validate   (InputValidateNode)
#     → retrieve         (RetrieveNode)
#     → rerank_filter    (RerankFilterNode)
#     → generate_answer  (GenerateAnswerNode)
#     → output_format    (OutputFormatNode)
#     → END
#
# Called by PolicyQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology — no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - All inner nodes declare required_trust_level = TrustLevel.ANONYMOUS
#   - get_output() designed together with PolicyQAGraphNode.merge_output()
#   - execute(self, state) canonical signature on every node — runtime config
#     reaches config-consuming nodes via constructor injection (self.config,
#     the {"configurable": {...}} dict from PolicyQAGraphNode._parent_config())
#   - The caller's input_context is seeded into the inner state by
#     _extra_initial_state() via the context bridge (GraphNode.execute() does
#     not forward it on subgraph.invoke(); see src/graph/context_bridge.py)
#   - No platform-SDK imports

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for GOV-C2-001.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by PolicyQAGraphNode.get_subgraph() in graph.py.

    Pipeline (linear RAG):
        START
          → input_validate   (InputValidateNode)
          → retrieve         (RetrieveNode)
          → rerank_filter    (RerankFilterNode)
          → generate_answer  (GenerateAnswerNode)
          → output_format    (OutputFormatNode)
          → END

    All nodes are FunctionNode subclasses with ANONYMOUS trust_level.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "gov_c2_001_policy_regulation_qa_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory config for the deterministic RAG inner graph."""
        pass

    # ── Initial state ─────────────────────────────────────────────────────────

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the inner graph's state with the caller's input_context.

        GraphNode.execute() does not forward input_context on subgraph.invoke()
        (SDK 1.0.1); PolicyQAGraphNode.extract_input() stashes it via the
        context bridge immediately before the inner invoke, and this hook
        (called by BaseGraph.invoke while building initial state) reads it
        back. Inner domain nodes keep their plain state["input_context"] reads.
        """
        return {"input_context": get_caller_input_context()}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.
        Every key registered here is referenced in add_edges().

        Config-consuming nodes (retrieve / rerank_filter / generate_answer)
        receive ``self.config`` — the graph-level config dict handed to this
        graph's constructor by PolicyQAGraphNode.get_subgraph()
        (``{"configurable": {top_k, score_threshold, ...}}`` built by
        _parent_config() from config/config.yaml) — via constructor injection.
        The execute() signature stays canonical: execute(self, state).
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode(config=self.config)
        self._nodes["rerank_filter"] = RerankFilterNode(config=self.config)
        self._nodes["generate_answer"] = GenerateAnswerNode(config=self.config)
        self._nodes["output_format"] = OutputFormatNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear RAG topology.

        Linear flow:
            input_validate → retrieve → rerank_filter → generate_answer
            → output_format → END.

        No conditional branching — the decline-on-low-confidence path is
        handled inside generate_answer (grounded flag), so all paths through
        the pipeline are linear.  route() satisfies the ABC but is not used at
        runtime.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by BaseGraph ABC.

        Linear topology; add_conditional_edges() is not used, so this method
        is never called at runtime.  Returns END on error so an unexpected
        invocation does not re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by PolicyQAGraphNode.merge_output() in graph.py
        as the `sub_result` argument.  Both methods are designed together to
        guarantee field-name consistency:

            Inner get_output() emits:   "answer_document", "answer",
                                        "citations", "filtered_docs",
                                        "grounded", "status"
            Outer merge_output() reads: sub_result.get(...) for each key above.
        """
        return {
            "answer_document": state.get("answer_document"),
            "answer": state.get("answer"),
            "citations": state.get("citations"),
            "filtered_docs": state.get("filtered_docs"),
            "grounded": state.get("grounded", False),
            "status": state.get("status"),
            # Carried explicitly: the boundary only moves the keys named here.
            "error_code": state.get("error_code"),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
