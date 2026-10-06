# GOV-C2-001 — Unit tests: the outer invoke() envelope (PolicyRegulationQAAgent.get_output)
#
# The envelope is the last thing between the graph state and the caller, and it
# is the second half of the output-gate contract. AgentBaseGraph.get_output()
# resolves its `output` key as `formatted_output or result` WITHOUT consulting
# status, and this template OVERRIDES get_output() — so an override that
# forwards `result` unconditionally re-implements that hole in template code
# and hands back, inside an error envelope, whatever the output gate refused.
#
# These call get_output() directly on a state dict so the resolution rule
# itself is pinned, independent of which node happened to produce that state:
# the envelope must be safe for ANY non-success state, not only for the one
# state today's PostProcessNode block happens to leave behind.

import json

from framework.schemas.agent_status import AgentStatus

from src.graph.graph import PolicyRegulationQAAgent

# What the inner RAG workflow produces before the output gate has run.
_PRE_GATE_DOCUMENT = (
    "========================================================================\n"
    "POLICY & REGULATION Q&A\n"
    "Answer:\n"
    "An administrative appeal must be filed within three months (Art. 18).\n"
    "  [GOV-KB-002] 行政不服審査法 Art. 18\n"
    "Grounded: YES"
)
_PRE_GATE_ANSWER = "An administrative appeal must be filed within three months (Art. 18)."
_PRE_GATE_CITATIONS = json.dumps([{"id": "GOV-KB-002", "source": "行政不服審査法 Art. 18"}], ensure_ascii=False)


def _state(**overrides) -> dict:
    state = {
        "status": AgentStatus.SUCCESS.value,
        "formatted_output": _PRE_GATE_DOCUMENT,
        "result": _PRE_GATE_DOCUMENT,
        "answer": _PRE_GATE_ANSWER,
        "answer_document": _PRE_GATE_DOCUMENT,
        "citations": _PRE_GATE_CITATIONS,
        "filtered_docs": '[{"id": "GOV-KB-002", "text": "internal candidate row"}]',
        "grounded": True,
        "trace_id": "envelope-test",
        "correlation_id": "envelope-test",
        "node_history": ["InitializeNode", "PreProcessNode", "PolicyQAGraphNode", "PostProcessNode"],
    }
    state.update(overrides)
    return state


def _envelope(**overrides) -> dict:
    return PolicyRegulationQAAgent().get_output(_state(**overrides))


class TestSuccessEnvelope:
    """The control. Without it every containment assertion below would also
    pass on an envelope that returns nothing at all."""

    def test_success_surfaces_the_gated_answer_and_the_structured_result(self):
        envelope = _envelope()
        assert envelope["status"] == AgentStatus.SUCCESS.value
        assert envelope["output"] == _PRE_GATE_DOCUMENT
        assert envelope["formatted_output"] == _PRE_GATE_DOCUMENT
        assert envelope["result"] == _PRE_GATE_DOCUMENT
        assert envelope["answer"] == _PRE_GATE_ANSWER
        assert envelope["citations"] == _PRE_GATE_CITATIONS
        assert envelope["filtered_docs"]
        assert envelope["grounded"] is True
        # The base envelope keys survive the extension.
        for key in ("trace_id", "correlation_id", "node_history"):
            assert key in envelope


class TestErrorEnvelopeContainment:
    def test_result_is_never_surfaced_on_a_non_success_outcome(self):
        """`result` is the value the gate either approved or replaced. On a
        non-success outcome it is not the caller's to see."""
        envelope = _envelope(status=AgentStatus.ERROR.value)
        assert envelope["result"] is None

    def test_the_or_result_fallback_is_dead_on_an_error(self):
        """The hole in the base envelope: with no `formatted_output`, `output`
        falls through to `result`. On a non-success outcome an absent gate
        output must STAY absent — it must never become the pre-gate answer.

        A falsy `formatted_output` does not suppress that fallback, it
        ACTIVATES it, which is why this case is pinned explicitly.
        """
        for falsy in (None, "", {}):
            envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=falsy)
            blob = json.dumps(envelope, ensure_ascii=False)
            assert not envelope["output"], falsy
            assert envelope["result"] is None, falsy
            assert _PRE_GATE_ANSWER not in blob, falsy
            assert "POLICY & REGULATION" not in blob, falsy
            assert "GOV-KB-002" not in blob, falsy

    def test_error_surfaces_the_gate_notice_and_nothing_else(self):
        """What the gate itself produced is the caller's whole error surface."""
        notice = "[ANSWER REDACTED: output contained a disallowed pattern (jwt). Contact the policy KB administrator.]"
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=notice, result=None)
        assert envelope["output"] == notice
        assert envelope["formatted_output"] == notice
        assert envelope["result"] is None

    def test_structured_domain_fields_are_withheld_on_an_error(self):
        envelope = _envelope(status=AgentStatus.ERROR.value)
        for key in ("answer", "citations", "filtered_docs", "grounded"):
            assert envelope[key] is None, key

    def test_the_pre_gate_document_is_never_a_surfaced_key(self):
        """`answer_document` is the raw pre-gate assembly. It must not appear
        in the envelope on either path — the gate's output is the contract."""
        for status in (AgentStatus.SUCCESS.value, AgentStatus.ERROR.value):
            assert "answer_document" not in _envelope(status=status), status

    def test_containment_holds_for_every_non_success_status(self):
        """Not an ERROR special case. Any status that is not SUCCESS means the
        output gate did not pass the response — including the terminal statuses
        that route straight to finalize without post_process running at all,
        which is exactly when `formatted_output` is absent and the base
        fallback would otherwise resolve to the pre-gate `result`.
        """
        for status in (
            AgentStatus.TIMEOUT.value,
            AgentStatus.CANCELLED.value,
            AgentStatus.RETRY.value,
            AgentStatus.PENDING.value,
        ):
            envelope = _envelope(status=status, formatted_output=None)
            blob = json.dumps(envelope, ensure_ascii=False)
            assert envelope["result"] is None, status
            assert not envelope["output"], status
            assert _PRE_GATE_ANSWER not in blob, status
