# GOV-C2-001 — Unit Tests: RAG domain nodes + graph wiring
#
# Real, non-stub unit tests. They import the REAL production modules and
# assert real behaviour (grounding / abstention discipline, the high GOV
# score_threshold, node trust levels, the output gate, and the Cat 2
# two-layer nested graph composition).
#
# Audit events are patched at the node MODULE level (not via a sys.modules
# stub, which would break the real `shared` package the framework loads at
# import time). Patch pattern per node:
#     monkeypatch.setattr("src.nodes.<mod>.emit_trace_event", lambda *a, **k: None)

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.security.credential_detector import detect_credentials
from framework.schemas.trust_level import TrustLevel
from src.schemas.state import from_json, to_json


# ── Shared fixtures / helpers ─────────────────────────────────────────────────

# A grounded policy question whose content terms fully cover GOV-KB-002
# (行政不服審査法 / Administrative Appeal Act) — RerankFilterNode clears the 0.78 bar.
GROUNDED_QUESTION = "What is the filing deadline period for an administrative appeal?"

# An off-domain question with zero KB overlap — forces the abstention path.
UNGROUNDED_QUESTION = "How do I bake a chocolate cake at home?"


def _passage(pid="GOV-KB-002", score=1.0):
    """A KB passage dict as produced by RetrieveNode."""
    return {
        "id": pid,
        "source": "行政不服審査法 (Administrative Appeal Act) Art. 18",
        "text": (
            "A request for administrative review (appeal) must in principle be "
            "filed within three months from the day following the disposition."
        ),
        "score": score,
    }


# ── PreProcessNode (outer pre_process, VERIFIED_EXTERNAL) ─────────────────────


class TestPreProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_valid_question_returns_success(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUESTION,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUESTION

    def test_whitespace_is_collapsed(self):
        result = self.node(
            {
                "user_input": "  What   is   the   rule?  ",
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == "What is the rule?"

    def test_enriched_context_carries_channel(self):
        result = self.node(
            {
                "user_input": GROUNDED_QUESTION,
                "input_context": {"channel": "web"},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        ctx = from_json(result["enriched_context"])
        assert ctx["source"] == "PolicyRegulationQAAgent"
        assert ctx["channel"] == "web"

    def test_empty_input_returns_error(self):
        result = self.node(
            {"user_input": "", "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("empty" in e for e in result["error_log"])

    def test_non_string_input_returns_error(self):
        result = self.node(
            {"user_input": 12345, "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")

    def test_over_length_question_returns_error(self):
        result = self.node(
            {"user_input": "a" * 2001, "input_context": {}, "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value}
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("exceeds" in e for e in result["error_log"])

    def test_trust_level_is_verified_external(self):
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_execute_signature_is_state_first(self):
        import inspect
        from src.nodes.pre_process_node import PreProcessNode

        params = list(inspect.signature(PreProcessNode.execute).parameters.keys())
        assert params[0] == "self" and params[1] == "state"
        assert "_invoke_impl" not in PreProcessNode.__dict__

    def test_execute_signature_is_canonical_on_every_node(self):
        # Canonical node contract: exactly execute(self, state) — no extra
        # parameters. Config reaches nodes via constructor injection.
        import inspect
        from src.nodes.generate_answer_node import GenerateAnswerNode
        from src.nodes.input_validate_node import InputValidateNode
        from src.nodes.main_node import MainNode
        from src.nodes.output_format_node import OutputFormatNode
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode
        from src.nodes.rerank_filter_node import RerankFilterNode
        from src.nodes.retrieve_node import RetrieveNode

        for node_cls in (
            PreProcessNode,
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
            PostProcessNode,
            MainNode,
        ):
            params = list(inspect.signature(node_cls.execute).parameters.keys())
            assert params == [
                "self",
                "state",
            ], f"{node_cls.__name__}.execute must be execute(self, state), got {params}"


# ── InputValidateNode (inner domain node 1, ANONYMOUS) ─────────────────────────


class TestInputValidateNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.input_validate_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def test_valid_input_builds_query(self):
        result = self.node({"validated_input": GROUNDED_QUESTION, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["query"] == GROUNDED_QUESTION

    def test_falls_back_to_user_input(self):
        result = self.node({"user_input": GROUNDED_QUESTION, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["query"] == GROUNDED_QUESTION

    def test_control_chars_are_stripped(self):
        result = self.node({"validated_input": "ab\x00cd\x07 ef", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "\x00" not in result["query"]
        assert result["query"] == "abcd ef"

    def test_empty_query_returns_error(self):
        result = self.node({"validated_input": "   ", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("empty" in e for e in result["error_log"])

    def test_too_short_query_returns_error(self):
        result = self.node({"validated_input": "ab", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("shorter" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RetrieveNode (inner domain node 2, ANONYMOUS) ──────────────────────────────


class TestRetrieveNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.retrieve_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.retrieve_node import RetrieveNode

        self.node = RetrieveNode()

    def test_retrieves_top_k_candidates(self):
        result = self.node({"query": GROUNDED_QUESTION, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["retrieved_count"] == 5  # default top_k
        docs = from_json(result["retrieved_docs"])
        # The appeal-deadline question ranks GOV-KB-002 first with a high score.
        assert docs[0]["id"] == "GOV-KB-002"
        assert docs[0]["score"] >= 0.78

    def test_top_k_config_limits_candidates(self):
        # Config flows via constructor injection:
        # register_nodes() passes the graph's {"configurable": {...}} dict.
        from src.nodes.retrieve_node import RetrieveNode

        node = RetrieveNode(config={"configurable": {"top_k": 2}})
        result = node.execute({"query": GROUNDED_QUESTION})
        assert result["retrieved_count"] == 2
        assert len(from_json(result["retrieved_docs"])) == 2

    def test_unrelated_query_scores_below_threshold(self):
        result = self.node({"query": UNGROUNDED_QUESTION, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        docs = from_json(result["retrieved_docs"])
        # No candidate clears the 0.78 grounding bar -> supports abstention downstream.
        assert all(d["score"] < 0.78 for d in docs)

    def test_missing_query_returns_error(self):
        result = self.node({"caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert any("missing" in e for e in result["error_log"])

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── RerankFilterNode (inner domain node 3, ANONYMOUS) ──────────────────────────


class TestRerankFilterNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.rerank_filter_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        self.node = RerankFilterNode()

    def test_high_score_passage_grounds(self):
        docs = to_json([_passage(score=1.0), _passage("GOV-KB-001", score=0.6)])
        result = self.node({"retrieved_docs": docs, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["grounded"] is True
        filtered = from_json(result["filtered_docs"])
        assert len(filtered) == 1
        assert filtered[0]["id"] == "GOV-KB-002"

    def test_below_threshold_is_not_grounded(self):
        docs = to_json([_passage("GOV-KB-001", score=0.6), _passage("GOV-KB-005", score=0.4)])
        result = self.node({"retrieved_docs": docs, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["grounded"] is False
        assert from_json(result["filtered_docs"]) == []

    def test_life_safety_floor_overrides_low_config_threshold(self):
        # A malicious/low config threshold (0.5) must be floored to 0.75 for GOV:
        # the 0.60 passage stays excluded; only the 0.76 passage grounds.
        # Config flows via constructor injection.
        from src.nodes.rerank_filter_node import RerankFilterNode

        docs = to_json([_passage("GOV-KB-002", score=0.76), _passage("GOV-KB-001", score=0.60)])
        node = RerankFilterNode(config={"configurable": {"score_threshold": 0.5}})
        result = node.execute({"retrieved_docs": docs})
        assert result["grounded"] is True
        filtered = from_json(result["filtered_docs"])
        assert len(filtered) == 1
        assert filtered[0]["id"] == "GOV-KB-002"

    def test_empty_candidates_not_grounded(self):
        result = self.node({"retrieved_docs": to_json([]), "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["grounded"] is False
        assert from_json(result["filtered_docs"]) == []

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── GenerateAnswerNode (inner domain node 4, ANONYMOUS) ────────────────────────


class TestGenerateAnswerNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.generate_answer_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.generate_answer_node import GenerateAnswerNode

        self.node = GenerateAnswerNode()

    def test_grounded_answer_cites_passages(self):
        state = {
            "query": GROUNDED_QUESTION,
            "filtered_docs": to_json([_passage()]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "GOV-KB-002" in result["answer"]
        citations = from_json(result["citations"])
        assert citations[0]["id"] == "GOV-KB-002"
        assert "Administrative Appeal Act" in citations[0]["source"]

    def test_declines_when_no_grounded_passages(self):
        from src.nodes.generate_answer_node import _DECLINE_MESSAGE

        state = {
            "query": UNGROUNDED_QUESTION,
            "filtered_docs": to_json([]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        # Grounding discipline: declines rather than fabricating regulatory guidance.
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["answer"] == _DECLINE_MESSAGE
        assert from_json(result["citations"]) == []

    def test_decline_message_is_explicit(self):
        from src.nodes.generate_answer_node import _DECLINE_MESSAGE

        assert "no answer is given" in _DECLINE_MESSAGE.lower()

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── OutputFormatNode (inner domain node 5, ANONYMOUS) ──────────────────────────


class TestOutputFormatNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.output_format_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.output_format_node import OutputFormatNode

        self.node = OutputFormatNode()

    def _grounded_state(self):
        return {
            "query": GROUNDED_QUESTION,
            "answer": "Based strictly on the retrieved sources, the appeal window is three months.",
            "citations": to_json([{"id": "GOV-KB-002", "source": "行政不服審査法 Art. 18"}]),
            "grounded": True,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }

    def test_assembles_cited_document(self):
        result = self.node(self._grounded_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        doc = result["answer_document"]
        assert result["result"] == doc
        assert "POLICY & REGULATION Q&A" in doc
        assert "GOV-KB-002" in doc
        assert "Grounded: YES" in doc

    def test_ungrounded_document_marks_no_grounding(self):
        state = {
            "query": UNGROUNDED_QUESTION,
            "answer": "No confident passage was found.",
            "citations": to_json([]),
            "grounded": False,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        doc = self.node(state)["answer_document"]
        assert "Grounded: NO" in doc
        assert "none" in doc.lower()

    def test_empty_answer_uses_fallback(self):
        state = {
            "query": GROUNDED_QUESTION,
            "answer": "",
            "citations": to_json([]),
            "grounded": False,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
        }
        result = self.node(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["answer_document"] is not None

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


# ── PostProcessNode (outer post_process, output gate, ANONYMOUS) ───────────────


class TestPostProcessNode:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    def test_clean_answer_passes_gate(self):
        doc = "POLICY & REGULATION Q&A\nAnswer: the appeal window is three months.\nGrounded: YES"
        result = self.node({"answer_document": doc, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == doc
        assert result["result"] == doc

    def test_empty_answer_uses_fallback(self):
        result = self.node({"answer_document": "", "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No answer content" in result["formatted_output"]

    def test_output_gate_redacts_credential_leak(self):
        leaky = "POLICY & REGULATION Q&A\ntoken=sk-abcdefghij0123456789ABCDEF"
        result = self.node({"answer_document": leaky, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        # Behaviour, not wording: the leak is blocked, the caller-facing
        # surface carries a NON-EMPTY withholding notice, and the error names
        # the gate class.
        #
        # Strengthened (containment): the previous form asserted
        # `formatted_output == result`, i.e. that the pre-gate slot still
        # carried a value on the refused path. The envelope resolves
        # `formatted_output or result` without consulting status, so pinning
        # the two together pinned the defect. `result` must be CLEARED, and the
        # replacement must stay truthy so the fallback cannot re-open the path
        # the gate just closed.
        assert result["status"] == AgentStatus.ERROR.value
        assert result["formatted_output"], "the replacement must be truthy — see the envelope fallback"
        assert "REDACTED" in result["formatted_output"]
        assert "sk-abcdefghij0123456789ABCDEF" not in json.dumps(result)
        assert result["result"] is None
        assert (result["formatted_output"] or result["result"]) is result["formatted_output"]
        assert any("credential" in e for e in result["error_log"])

    def test_blocked_state_fields_redacted_verbatim(self):
        # Independent layer: a raw internal JSON blob embedded verbatim in the
        # document is replaced with [REDACTED] even when credential-free.
        blob = '[{"id": "GOV-KB-002", "score": 1.0, "text": "internal candidate row"}]'
        doc = f"POLICY & REGULATION Q&A\nDebug dump: {blob}\nGrounded: YES"
        result = self.node(
            {
                "answer_document": doc,
                "retrieved_docs": blob,
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert blob not in result["formatted_output"]
        assert "[REDACTED]" in result["formatted_output"]

    def test_statutory_content_passes_through_byte_identical(self):
        # The gate must not rewrite regulatory content: deadlines, article
        # numbers, years, thresholds and quoted statutory wording are the
        # product and pass through unchanged.
        doc = (
            "POLICY & REGULATION Q&A\n"
            "An appeal must be filed within three months (Art. 18). A "
            "disclosure decision is due within thirty days. Score bar 0.78; "
            "guideline year 2026; fee 300 units per application.\n"
            "Grounded: YES"
        )
        result = self.node({"answer_document": doc, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == doc

    def test_security_gate_output_helper_detects_and_clears(self):
        from src.nodes.post_process_node import _security_gate_output

        assert _security_gate_output("sk-abcdefghij0123456789ABCDEF") is not None
        assert _security_gate_output("Bearer abcdefgh12345678") is not None
        assert _security_gate_output("password = supersecret123") is not None
        assert _security_gate_output("A perfectly clean policy answer document.") is None

    def test_trust_level_anonymous(self):
        assert self.node.required_trust_level == TrustLevel.ANONYMOUS


class TestOutputGateDetectorParity:
    """The gate must never be NARROWER than the framework's own recognizer.

    The framework scans every node result with detect_credentials() and RAISES
    on a hit; BaseNode.__call__ then replaces the node's whole return with a
    bare error partial — discarding the containment the gate wrote. So a shape
    the framework catches and this gate misses is not a lenient gate, it is a
    containment BYPASS: the refused value survives in state with nothing
    cleared. The parametrization below covers every family the framework
    refuses, with the assertion above it keeping the probes honest.

    InputValidateNode imports the same helper for its caller-corpus ingest
    check, so this parity is also the ingest contract.
    """

    @pytest.mark.parametrize(
        "shape",
        [
            "sk_live_" + "a" * 20,  # stripe secret key
            "sk_test_" + "b" * 20,  # stripe test key
            "sk-" + "c" * 24,  # openai / generic api key
            "eyJ" + "d" * 20,  # JWT — header segment alone, NO dots
            "AKIA" + "EFGH1234IJKL5678",  # aws iam access key id
            "Bearer " + "e" * 24,  # http bearer token
            "postgresql://" + "kbuser:pw12345678@policy-db.internal:5432/gov",  # conn string
            "redis://" + "cache.internal:6379/0?password=hunter22",  # conn string
        ],
    )
    def test_gate_refuses_every_shape_the_framework_would_refuse(self, shape):
        from src.nodes.post_process_node import _security_gate_output

        # Keeps the parametrization honest: each probe really is a shape the
        # framework itself refuses one layer up.
        assert detect_credentials(shape), "probe shape is not a credential the framework refuses"
        violation = _security_gate_output(f"POLICY & REGULATION Q&A\nAnswer: reference {shape}")
        assert violation is not None, "gate is narrower than the framework — containment bypass"
        # The TYPE is named; the matched value is never returned. Echoing it
        # would put the refused string back into the node's own result, where
        # the framework scan raises and discards the clearing.
        assert shape not in violation

    def test_ordinary_statutory_content_is_not_flagged(self):
        """The other direction — a refuse-everything gate would pass the test
        above while destroying the product."""
        from src.nodes.post_process_node import _security_gate_output

        assert (
            _security_gate_output(
                "POLICY & REGULATION Q&A\n"
                "An appeal must be filed within three months (Art. 18). A disclosure "
                "decision is due within thirty days. Score bar 0.78; guideline year 2026.\n"
                "  [GOV-KB-002] 行政不服審査法 Art. 18\n"
                "Grounded: YES"
            )
            is None
        )


class TestOutputGateContainment:
    """Blocking a response must also CONTAIN it.

    AgentBaseGraph resolves the caller-facing value as `formatted_output or
    result` with no regard for status, and PolicyQAGraphNode.merge_output()
    has already copied the pre-gate answer, citations and filtered candidates
    into outer state. A gate that returns ERROR while leaving those populated
    still ships the refused answer inside the error envelope. These pin the
    clearing itself; the end-to-end consequence is pinned on the real /invoke
    surface in tests/proof_of_boundary/test_invoke_e2e.py.
    """

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    _LEAKY_DOC = (
        "========================================================================\n"
        "POLICY & REGULATION Q&A\n"
        "Answer:\n"
        "An administrative appeal must be filed within three months. token=sk-abcdefghij0123456789ABCDEF\n"
        "  [GOV-KB-002] 行政不服審査法 Art. 18\n"
        "Grounded: YES"
    )

    def _blocked(self):
        return self.node(
            {
                "answer_document": self._LEAKY_DOC,
                "answer": "An administrative appeal must be filed within three months.",
                "citations": to_json([{"id": "GOV-KB-002", "source": "行政不服審査法 Art. 18"}]),
                "filtered_docs": to_json([{"id": "GOV-KB-002", "text": "internal candidate row"}]),
                "grounded": True,
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )

    def test_block_clears_every_output_bearing_field(self):
        from src.nodes.post_process_node import _CONTENT_FIELDS

        result = self._blocked()
        assert result["status"] == AgentStatus.ERROR.value
        for field in _CONTENT_FIELDS:
            assert result[field] is None, f"{field} still carries its pre-gate value"

    def test_inert_provenance_is_pinned_content_free(self):
        """`grounded` is a bare boolean, so it may stay — but it is pinned
        False, never left True, so a refusal cannot be read as a grounded
        success."""
        assert self._blocked()["grounded"] is False

    def test_the_clearing_inventory_covers_every_merged_domain_field(self):
        """Inventory guard.

        The clearing set is only correct relative to what merge_output writes
        into outer state. A new domain field added there must be classified —
        cleared as content, or explicitly declared inert — and cannot quietly
        join the state and be released on a refused response.
        """
        from src.graph.graph import PolicyQAGraphNode
        from src.nodes.post_process_node import _CONTENT_FIELDS, _INERT_FIELDS

        merged = set(
            PolicyQAGraphNode().merge_output(
                {},
                {
                    "answer_document": "d",
                    "answer": "a",
                    "citations": "[]",
                    "filtered_docs": "[]",
                    "grounded": True,
                    "status": AgentStatus.SUCCESS.value,
                },
            )
        )
        classified = set(_CONTENT_FIELDS) | set(_INERT_FIELDS)
        # `status` is the control field the gate sets itself, not payload.
        assert merged - {"status"} <= classified, f"unclassified domain field(s): {merged - {'status'} - classified}"
        # `result` is this node's own output slot and must always be cleared.
        assert "result" in _CONTENT_FIELDS

    def test_the_replacement_defeats_the_envelope_fallback(self):
        """`formatted_output or result` must resolve to the withholding notice.

        An empty replacement ("" or {}) is falsy and hands the resolution
        straight back to `result` — the exact hole the clearing closes.
        """
        result = self._blocked()
        assert (result["formatted_output"] or result["result"]) is result["formatted_output"]
        assert result["formatted_output"]

    def test_block_releases_no_domain_content(self):
        result = self._blocked()
        blob = json.dumps(result, ensure_ascii=False)
        assert "sk-abcdefghij0123456789ABCDEF" not in blob
        assert "three months" not in blob
        assert "GOV-KB-002" not in blob
        assert "POLICY & REGULATION" not in blob

    def test_clearing_survives_the_framework_output_scan(self):
        """The violation names the pattern TYPE only, never the matched value.

        Echoing it would put the refused string back into this node's own
        result, where the framework's output-side credential scan RAISES — and
        a raise makes __call__ discard the whole return, so the clearing above
        would never be applied and the pre-gate values would stay in state.
        Driven through node(state) so that scan really runs.
        """
        result = self._blocked()
        blob = json.dumps(result, ensure_ascii=False)
        assert result["status"] == AgentStatus.ERROR.value
        assert "Traceback" not in blob, "the framework scan discarded the clearing return"
        assert result["result"] is None
        assert result["formatted_output"]

    def test_a_framework_only_shape_is_blocked_with_the_clearing_intact(self):
        """The bypass case that motivates detector parity: a shape the OLD
        local pattern set missed but the framework refuses. The gate must
        block it here — where the clearing is applied — rather than let the
        framework raise one layer up and discard the containment."""
        doc = self._LEAKY_DOC.replace("sk-abcdefghij0123456789ABCDEF", "AKIAEFGH1234IJKL5678")
        result = self.node({"answer_document": doc, "caller_trust_level": TrustLevel.ANONYMOUS.value})
        assert result["status"] == AgentStatus.ERROR.value
        assert result["result"] is None
        assert result["formatted_output"]
        assert "AKIAEFGH1234IJKL5678" not in json.dumps(result)


# ── Outer graph: AgentBaseGraph backbone (Cat 2 nested) ────────────────────────


class TestOuterGraphComposition:
    def test_registers_five_backbone_slots(self):
        from src.graph.graph import PolicyQAGraphNode, PolicyRegulationQAAgent
        from src.nodes.post_process_node import PostProcessNode
        from src.nodes.pre_process_node import PreProcessNode

        agent = PolicyRegulationQAAgent()
        agent.compile()
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], PolicyQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.graph import PolicyRegulationQAAgent

        agent = PolicyRegulationQAAgent()
        assert agent.name == "PolicyRegulationQAAgent"
        assert agent.state_schema is State

    def test_graph_alias_matches_real_class(self):
        from src.graph.graph import Graph, PolicyRegulationQAAgent

        assert Graph is PolicyRegulationQAAgent

    def test_main_slot_graphnode_contracts(self):
        from src.graph.graph import PolicyQAGraphNode

        node = PolicyQAGraphNode()
        assert node.error_strategy == "propagate"
        assert node.propagate_hitl is False
        # extract_input prefers validated_input, falls back to user_input
        assert node.extract_input({"validated_input": "V", "user_input": "U"}) == "V"
        assert node.extract_input({"user_input": "U"}) == "U"

    def test_merge_output_maps_subresult_keys(self):
        from src.graph.graph import PolicyQAGraphNode

        node = PolicyQAGraphNode()
        sub_result = {
            "answer_document": "DOC",
            "answer": "A",
            "citations": "[]",
            "filtered_docs": "[]",
            "grounded": True,
            "status": AgentStatus.SUCCESS.value,
            "node_history": ["x"],  # not forwarded by merge_output
        }
        delta = node.merge_output({}, sub_result)
        assert delta["answer_document"] == "DOC"
        assert delta["grounded"] is True
        assert delta["status"] == AgentStatus.SUCCESS.value
        assert set(delta.keys()) == {
            "answer_document",
            "answer",
            "citations",
            "filtered_docs",
            "grounded",
            "status",
            # Always forwarded so a reason can never be dropped at the boundary.
            "error_code",
        }

    def test_merge_output_defaults_grounded_false(self):
        from src.graph.graph import PolicyQAGraphNode

        node = PolicyQAGraphNode()
        delta = node.merge_output({}, {"answer_document": "DOC"})
        assert delta["grounded"] is False


# ── Inner graph: BaseGraph RAG topology ────────────────────────────────────────


class TestInnerDomainGraph:
    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        for mod in (
            "input_validate_node",
            "retrieve_node",
            "rerank_filter_node",
            "generate_answer_node",
            "output_format_node",
        ):
            monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)

    def test_registers_five_domain_nodes(self):
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.register_nodes()
        assert set(g._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }

    def test_name_and_state_schema(self):
        from src.schemas.state import State
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        assert g.name == "gov_c2_001_policy_regulation_qa_workflow"
        assert g.state_schema is State

    def test_inner_graph_invoke_grounds_and_cites(self):
        """Standalone inner-graph invoke (ANONYMOUS caller) runs the linear RAG
        pipeline and shapes the get_output() dict consumed by the outer merge_output()."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(GROUNDED_QUESTION, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["grounded"] is True
        assert "POLICY & REGULATION Q&A" in result["answer_document"]
        assert "GOV-KB-002" in result["answer_document"]

    def test_inner_graph_invoke_declines_when_ungrounded(self):
        """Abstention path: an off-domain question grounds nothing above the high
        GOV threshold, so the pipeline declines instead of fabricating guidance."""
        from framework.schemas.invocation_context import InvocationContext, TrustLevel
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        g = DomainWorkflowGraph()
        g.compile()
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = g.invoke(UNGROUNDED_QUESTION, ctx=ctx)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["grounded"] is False
        assert "Grounded: NO" in result["answer_document"]


# ── Trust gate (PreProcessNode — the only VERIFIED_EXTERNAL node) ──────────────


class TestTrustGate:
    """Trust-gate coverage for the outer pre_process slot.

    PreProcessNode is the only node in this template requiring VERIFIED_EXTERNAL
    trust. These tests invoke it through __call__ (the real BaseNode entry
    point) so the trust gate actually runs — a direct execute() call would
    bypass it. On denial the framework RETURNS an error dict; it never raises.
    """

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.pre_process_node.emit_trace_event", lambda *a, **k: None)

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def test_trust_gate_rejects_untrusted_caller_before_execute(self):
        # ANONYMOUS < required VERIFIED_EXTERNAL: the trust gate in __call__
        # denies BEFORE execute() runs and RETURNS an error dict (no exception
        # raised).
        result = self.node(
            {
                "user_input": GROUNDED_QUESTION,
                "input_context": {},
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert "trust gate denied" in " ".join(result["error_log"])
        # execute() never ran, so its output key is absent.
        assert "validated_input" not in result

    def test_trust_gate_admits_trusted_caller(self):
        # VERIFIED_EXTERNAL caller clears the gate; execute() runs and succeeds.
        result = self.node(
            {
                "user_input": GROUNDED_QUESTION,
                "input_context": {},
                "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["validated_input"] == GROUNDED_QUESTION
