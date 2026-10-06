"""AgentCore Platform v1.0"""

# GOV-C2-001 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — identical to Cat 1, do NOT override add_edges()):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                             ↓ (RETRY, max 3)
#                                          pre_process
#
#   `main` slot is a GraphNode subclass (PolicyQAGraphNode) that delegates the
#   full RAG domain workflow to DomainWorkflowGraph (inner BaseGraph).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (RAG topology)
#   src/graph/context_bridge.py        ← input_context hand-off outer → inner
#
# Invariants:
#   - PolicyRegulationQAAgent inherits AgentBaseGraph (framework base class)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - PolicyQAGraphNode assigned to self._nodes["main"]
#   - PreProcessNode (VERIFIED_EXTERNAL) in pre_process slot (input trust gate)
#   - PostProcessNode (ANONYMOUS) in post_process slot (output gate)
#   - merge_output() returns only changed keys
#   - class name matches config/agent.yaml `class:` field exactly
#   - add_edges() NOT overridden on the outer graph
#   - no platform-SDK imports

from pathlib import Path
from typing import Any, ClassVar

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

# Runtime-parameter file: src/graph/graph.py -> parents[2] is the repo root.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _runtime_config() -> dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    config/config.yaml (separate from the static manifest config/agent.yaml)
    declares the runtime settings: max_retry / timeout_s at the root, plus the
    llm and retrieval blocks the inner RAG pipeline consumes. Returns an empty
    dict — never raises — when the file is absent, unreadable, not valid YAML,
    or not a mapping; the inner nodes then fall back to their declared
    defaults. PyYAML is loaded lazily — it is a framework runtime dependency,
    so importing it on demand avoids a hard module-load coupling.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


class PolicyQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of PolicyRegulationQAAgent.

    Wraps DomainWorkflowGraph (inner Cat 2 RAG BaseGraph).
    Called by AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()  — instantiate and return DomainWorkflowGraph
      extract_input() — pull validated_input from outer state; bridge input_context
      merge_output()  — map sub_result fields into outer state delta (changed keys only)
      error_strategy  — "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    def __init__(self, runtime_config: dict[str, Any] | None = None) -> None:
        """Receive the runtime config from the outer graph.

        A BaseNode has no config back-reference of its own, so the outer
        AgentBaseGraph reads `self.config` and threads it in here at
        register_nodes() time. Static construction input - not mutable state.
        """
        # A non-mapping runtime config degrades to {} instead of raising: reading and
        # parsing config/config.yaml belongs to the entry point, and this node only has
        # to survive whatever it is handed.
        self._runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time and to match the Cat 2 pattern.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined upstream has no validated input to work on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates and normalises the raw user_input and writes
        the result to validated_input. Prefer that; fall back to user_input if
        validated_input is absent (e.g. in unit tests).

        Also bridges the caller's input_context to the inner graph:
        GraphNode.execute() does not forward input_context on subgraph.invoke()
        (SDK 1.0.1), and extract_input is the last template-code hook that sees
        the outer state before the inner invoke — see
        src/graph/context_bridge.py.
        """
        set_caller_input_context(state.get("input_context") or {})
        result = state.get("validated_input", state.get("user_input", ""))
        return str(result) if result is not None else ""

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  → "answer_document", "answer", "citations",
                                       "filtered_docs", "grounded", "status"
          This merge_output() reads → sub_result.get(...) for each of these keys.

        PostProcessNode (outer post_process) reads answer_document from state to
        apply the output gate and set formatted_output.
        """
        return {
            "answer_document": sub_result.get("answer_document"),
            "answer": sub_result.get("answer"),
            "citations": sub_result.get("citations"),
            "filtered_docs": sub_result.get("filtered_docs"),
            "grounded": sub_result.get("grounded", False),
            "status": sub_result.get("status"),
            # Outer reason wins: a reason settled before the inner run is the
            # real one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
        }

    def _parent_config(self) -> dict[str, Any]:
        """Forward the declared runtime settings to the inner graph.

        Reads config/config.yaml (see _runtime_config) and exposes the declared
        settings under the ``configurable`` key — the shape the config-consuming
        inner nodes read from their constructor-injected config
        (DomainWorkflowGraph.register_nodes() passes the graph's ``self.config``
        — this dict — into the config-consuming node constructors). Without
        this, the declared retrieval settings (``top_k``, ``score_threshold``)
        would be dead configuration text and every inner node would silently
        fall back to its hard-coded default. Only keys the file actually
        declares (non-None) are forwarded; absent keys fall back to each node's
        default.

        The LLM settings (``temperature`` / ``max_tokens`` /
        ``system_prompt_template``) are forwarded for the production LLM wiring
        point. Generation in this template is deterministic (extractive), so
        those keys are declared-but-inert until a real LLM is wired in; they
        are never used to fake an LLM call.
        """
        cfg = self._runtime_config
        llm = cfg.get("llm", {}) or {}
        retrieval = cfg.get("retrieval", {}) or {}
        if not isinstance(llm, dict):
            llm = {}
        if not isinstance(retrieval, dict):
            retrieval = {}
        declared = {
            "top_k": retrieval.get("top_k"),
            "score_threshold": retrieval.get("score_threshold"),
            "system_prompt_template": llm.get("system_prompt_template"),
            "temperature": llm.get("temperature"),
            "max_tokens": llm.get("max_tokens"),
        }
        return {"configurable": {k: v for k, v in declared.items() if v is not None}}


class PolicyRegulationQAAgent(AgentBaseGraph):
    """Outer graph for GOV-C2-001 (Cat 2 — RAGAgent).

    Inherits AgentBaseGraph directly (framework base class). Domain logic is fully
    encapsulated in PolicyQAGraphNode (main slot), which delegates to
    DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed — identical to Cat 1):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() and get_output() are the only overrides:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process: PreProcessNode    (VERIFIED_EXTERNAL — input trust gate)
      - main:        PolicyQAGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode  (ANONYMOUS — output gate)
      - get_output(): surfaces the structured domain result on success

    add_edges() is NOT overridden — backbone wiring belongs to the framework.

    Class name MUST match config/agent.yaml `class:` field exactly.
    server.py imports this as `Graph` via the alias below.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "PolicyRegulationQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = PolicyQAGraphNode(runtime_config=self.config)
        self._nodes["post_process"] = PostProcessNode()

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Surface the domain policy-Q&A result on the outer invoke() return.

        AgentBaseGraph.get_output() returns only the minimal
        ``{output, status, trace_id, correlation_id, node_history}`` envelope.
        On the compiled outer-graph success path that would drop the structured
        domain result the inner DomainWorkflowGraph produces (merged into outer
        state by PolicyQAGraphNode.merge_output(): answer, citations,
        filtered_docs, grounded) — every one of them would be None to a caller
        of ``agent.invoke()`` even on a successful, grounded answer. This
        override extends the base envelope so a successful invocation actually
        returns the structured answer.

        Output-gate invariant preserved (fail-closed):
          * ``formatted_output`` is what the gated PostProcessNode produced, so
            it is the only caller-facing value on either path (on a block it is
            the gate's own content-free withholding notice). Never the pre-gate
            raw ``state["answer_document"]``.
          * ``result`` is surfaced ONLY on the gated success path. The base
            envelope resolves its ``output`` key as
            ``formatted_output or result`` WITHOUT consulting status, so on a
            non-success outcome that fallback is re-resolved here as well: an
            absent gate output stays absent and never becomes the pre-gate
            answer. Without both halves an error envelope would carry the very
            response the output gate refused — and a falsy ``formatted_output``
            would not suppress the fallback, it would ACTIVATE it.
          * The structured fields (answer / citations / filtered_docs /
            grounded) are surfaced ONLY when the gate passed
            (status == SUCCESS). On any non-success outcome — including a
            credential block — they are withheld (None).
        """
        output = dict(super().get_output(state))
        # A run that completed WITHOUT carrying out the request holds the
        # sentence saying what to correct, not a domain product: none of the
        # structured fields below were produced, so none of them is released.
        # The base envelope already holds that sentence. SUCCESS here reports
        # that the run reached a defined end safely, not that the request was
        # carried out.
        if state.get("error_code"):
            return output

        succeeded = state.get("status") == AgentStatus.SUCCESS.value

        formatted_output = state.get("formatted_output")
        output["formatted_output"] = formatted_output
        if succeeded:
            output["result"] = state.get("result")
        else:
            output["result"] = None
            # Re-resolve the base envelope's value without the `or result`
            # fallback: on a non-success outcome the gate's own output is all
            # the caller may see, and an absent one stays absent.
            output["output"] = formatted_output or None

        # Structured domain result — surfaced only on the gated success path.
        output["answer"] = state.get("answer") if succeeded else None
        output["citations"] = state.get("citations") if succeeded else None
        output["filtered_docs"] = state.get("filtered_docs") if succeeded else None
        output["grounded"] = state.get("grounded") if succeeded else None
        return output

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Alias for backward compat (server.py imports Graph).
# Class name PolicyRegulationQAAgent matches config/agent.yaml class: field.
Graph = PolicyRegulationQAAgent
