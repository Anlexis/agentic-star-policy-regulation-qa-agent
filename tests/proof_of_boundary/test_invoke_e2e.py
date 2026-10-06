# PB: End-to-end business behaviour through POST /invoke — src/api/server.py
#
# Proves the supported caller-data contract produces REAL outcomes through the
# full nested graph (outer backbone → inner domain pipeline), not only the
# built-in baseline:
#   - a grounded, cited answer computed from a CALLER-supplied policy corpus
#     (which also proves input_context crosses the outer→inner graph boundary
#     via the context bridge — the framework does not forward it)
#   - the built-in-corpus baseline still answers real questions when the
#     caller supplies no data
#   - the abstention path (off-domain question → decline, never a guess)
#   - validation rejections for malformed caller data, incl. non-finite
#     numerics arriving as strings AND as raw JSON NaN/Infinity
#   - the adapter-level size cap and the Bearer-auth boundary
#   - an output-schema scan: the caller-facing document renders only validated
#     corpus content + citations — no internal JSON state, no credentials
#
# The app is driven through its real ASGI interface: every request crosses the
# entry-point auth, the outer trust/input gates, the input_context bridge into
# the inner graph, all five domain nodes, and the output gate.

import asyncio
import json

import pytest

from src.api import server as server_module  # noqa: F401  (import = boot check)
from src.api.server import app

_TOKEN = "pb-invoke-e2e-token"

# Fully covered by the caller passage below (every query content-token appears
# in its text or keywords) → retrieval score 1.0 ≥ the 0.78 grounding bar.
_CALLER_QUESTION = "What is the deadline in days for a bicycle parking permit application?"

_CALLER_PASSAGES = [
    {
        "id": "ward_ordinance_012",
        "source": "Municipal Ordinance No. 12 Art. 4",
        "text": (
            "Bicycle parking permits are issued within fourteen days of a " "complete application to the ward office."
        ),
        "keywords": ["bicycle", "parking", "permit", "application", "days", "deadline"],
    }
]

# Grounds on the built-in sample corpus (score 1.0 on the appeal passage).
_BUILTIN_QUESTION = "What is the filing deadline period for an administrative appeal?"

# Zero overlap with either corpus — forces the abstention path.
_OFF_DOMAIN_QUESTION = "How do I bake a chocolate cake at home?"


def _post_invoke(payload: dict, with_auth: bool = True) -> tuple[int, dict, str]:
    """POST /invoke through the real ASGI app; returns (status, body, raw_text)."""
    body = json.dumps(payload).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if with_auth:
        headers.append((b"authorization", f"Bearer {_TOKEN}".encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/invoke",
        "raw_path": b"/invoke",
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    messages = []
    sent = {"body": b""}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    raw = sent["body"].decode()
    return start["status"], json.loads(raw or "{}"), raw


@pytest.fixture(autouse=True)
def token_configured(monkeypatch):
    """Deploy-shaped server environment: INVOKE_AUTH_TOKEN set, caller uses Bearer."""
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)


def _invoke(question: str, input_context: dict | None = None) -> dict:
    status_code, body, _ = _post_invoke(
        {"input": question, "session_id": "pb-invoke-e2e", "input_context": input_context or {}}
    )
    assert status_code == 200, f"expected 200, got {status_code}: {body}"
    return body


class TestInvokeEndToEnd:
    def test_caller_corpus_produces_grounded_cited_answer(self):
        """Caller passages must ground a real answer through the full nested path."""
        body = _invoke(_CALLER_QUESTION, {"policy_passages": _CALLER_PASSAGES})

        assert body["status"] == "success"
        output = body["output"]
        assert "Grounded: YES" in output, output
        assert "ward_ordinance_012" in output
        assert "Municipal Ordinance No. 12 Art. 4" in output
        assert "fourteen days" in output
        assert "no answer is given" not in output.lower()
        assert body["grounded"] is True

    def test_builtin_baseline_still_answers_without_caller_data(self):
        """Absent caller data degrades to the built-in corpus — still real work."""
        body = _invoke(_BUILTIN_QUESTION)

        assert body["status"] == "success"
        assert "Grounded: YES" in body["output"]
        assert "GOV-KB-002" in body["output"]

    def test_off_domain_question_declines(self):
        """Abstention discipline end-to-end: no grounded passage → decline."""
        body = _invoke(_OFF_DOMAIN_QUESTION)

        assert body["status"] == "success"
        assert "Grounded: NO" in body["output"]
        assert "no answer is given" in body["output"].lower() or "does not contain" in body["output"].lower()

    def test_caller_threshold_tightens_grounding_end_to_end(self):
        """A caller score_threshold override must tighten the bar through /invoke."""
        # The built-in probe question scores 5/6 ≈ 0.83 on its best passage:
        # grounded at the shipped 0.78 bar, declined at a caller 0.90 bar.
        probe = "What is the official filing deadline period for an administrative appeal?"
        grounded = _invoke(probe)
        assert "Grounded: YES" in grounded["output"]

        declined = _invoke(probe, {"retrieval": {"score_threshold": 0.90}})
        assert declined["status"] == "success"
        assert "Grounded: NO" in declined["output"]

    def test_invalid_caller_passage_is_rejected_without_echo(self):
        """Malformed caller data must produce a validation error naming the
        field — and the rejected value must never round-trip into the response."""
        bad_id = "INVALID UPPER ID WITH SPACES"
        status_code, body, raw = _post_invoke(
            {
                "input": _CALLER_QUESTION,
                "session_id": "pb-invoke-e2e",
                "input_context": {"policy_passages": [dict(_CALLER_PASSAGES[0], id=bad_id)]},
            }
        )
        assert status_code == 200
        # Declined rather than terminated: the caller can correct the value and
        # send the request again on the same conversation.
        assert body["status"] == "success", body
        assert "could not be accepted" in body["output"], body
        # No answer is produced, and the rejected value never round-trips.
        assert body.get("answer") in (None, ""), body
        assert bad_id not in raw

    @pytest.mark.parametrize(
        "bad_threshold",
        ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), float("-inf")],
        ids=["str-nan", "str-inf", "str-neginf", "raw-nan", "raw-inf", "raw-neginf"],
    )
    def test_non_finite_threshold_fails_closed_not_open(self, bad_threshold):
        """A NaN/Infinity score_threshold must ERROR with no answer — never a
        'success with the bar silently disabled' fail-open (raw floats cover
        Python json's bare-NaN extension reaching the request body)."""
        body = _invoke(
            _BUILTIN_QUESTION,
            {"retrieval": {"score_threshold": bad_threshold}},
        )
        assert body["status"] == "success", body
        # Declined with the reason - never a "success with the bar silently
        # disabled" fail-open.
        assert "could not be accepted" in body["output"], body
        assert body.get("answer") in (None, ""), body

    def test_oversized_input_context_is_capped_at_the_adapter(self):
        """The adapter rejects oversized input_context before it reaches the graph."""
        status_code, body, _ = _post_invoke(
            {
                "input": _CALLER_QUESTION,
                "session_id": "pb-invoke-e2e",
                "input_context": {"policy_passages": [{"id": "big", "text": "a" * 300_000}]},
            }
        )
        assert status_code == 413

    def test_missing_bearer_token_is_unauthorized(self):
        status_code, body, _ = _post_invoke(
            {"input": _BUILTIN_QUESTION, "session_id": "pb-invoke-e2e"},
            with_auth=False,
        )
        assert status_code == 401

    def test_output_schema_scan_on_grounded_response(self):
        """The caller-facing document renders only validated corpus content and
        citations — no internal pipeline state, no credential material."""
        body = _invoke(_CALLER_QUESTION, {"policy_passages": _CALLER_PASSAGES})
        output = body["output"]

        # No raw internal JSON state in the external surface.
        assert '"keywords"' not in output
        assert '"score"' not in output
        assert '"retrieved_docs"' not in output
        # Citations name only the validated corpus.
        assert "ward_ordinance_012" in output
        assert "GOV-KB" not in output  # built-in corpus replaced, not mixed
        # Credential scan (same patterns the output gate enforces).
        from src.nodes.post_process_node import _security_gate_output

        assert _security_gate_output(output) is None


class TestBlockedOutputIsContained:
    """A blocked response must not ship the answer it blocked.

    AgentBaseGraph resolves the caller-facing value as `formatted_output or
    result` WITHOUT consulting status, and this template overrides get_output()
    — so an output gate that only flips the status still returns the refused
    answer inside the error envelope, and an override that forwards `result`
    unconditionally re-opens the same hole in template code. Exercised on the
    real /invoke surface, with the clean-path control beside it: a green
    containment result on a request that never produced anything would prove
    nothing.

    The fault is injected on the DATA path, never on the gate: the caller asks
    a question that legitimately trips the domain output ceiling. The probe
    shape below is deliberately one the FRAMEWORK does not refuse (`sk-` with
    18 characters — the framework recognizer needs 20+), so the request reaches
    the domain gate instead of being refused a layer earlier; the parity of the
    two recognizers is pinned separately in tests/unit/test_nodes.py.
    """

    # Trips the domain gate (sk- + 16 or more) but NOT the framework detector
    # (sk- + 20 or more), so the domain gate is genuinely the component under
    # test rather than a framework raise one node upstream.
    _DOMAIN_ONLY_SHAPE = "sk-" + "a" * 18
    _GATED_QUESTION = (
        "What is the deadline in days for a bicycle parking permit application, "
        "cross-referenced against key " + _DOMAIN_ONLY_SHAPE + "?"
    )

    def test_clean_path_control_releases_the_answer(self):
        """Control: the same shaped request, unblocked, really does produce the
        grounded cited answer the blocked run must withhold."""
        body = _invoke(_CALLER_QUESTION, {"policy_passages": _CALLER_PASSAGES})
        assert body["status"] == "success"
        assert "Grounded: YES" in body["output"]
        assert "fourteen days" in body["output"]
        assert body["result"] is not None
        assert body["answer"] is not None
        assert body["grounded"] is True

    def test_blocked_output_is_not_released_through_the_envelope(self):
        status_code, body, raw = _post_invoke(
            {
                "input": self._GATED_QUESTION,
                "session_id": "pb-invoke-e2e",
                "input_context": {"policy_passages": _CALLER_PASSAGES},
            }
        )
        assert status_code == 200
        assert body["status"] == "error", body

        # The block happened AT the output gate, not somewhere upstream: the
        # request really reached post_process and was refused there.
        assert "PostProcessNode" in body["node_history"], body["node_history"]

        # The gate's own notice is the whole caller-facing surface, and it is
        # non-empty so the `formatted_output or result` fallback resolves to it.
        assert body["output"]
        assert "REDACTED" in body["output"]

        # Nothing pre-gate survives the envelope.
        assert body["result"] is None
        for key in ("answer", "citations", "filtered_docs", "grounded"):
            assert body[key] is None, f"{key} released on the error path"

        # No released text anywhere in the body: no assembled document, no
        # corpus content, no citation label, and no echo of the caller string
        # the gate refused.
        assert "POLICY & REGULATION Q&A" not in raw
        assert "fourteen days" not in raw
        assert "ward_ordinance_012" not in raw
        assert "Municipal Ordinance" not in raw
        assert self._DOMAIN_ONLY_SHAPE not in raw

        # No traceback and no source paths: a violation message quoting the
        # refused value would trip the framework scan on this very result, and
        # the framework replaces a raising node's return with a traceback —
        # discarding the containment along with it.
        assert "Traceback" not in raw
        assert "src/nodes/" not in raw

    def test_framework_shaped_caller_data_releases_nothing(self):
        """The unpatched companion: a shape the FRAMEWORK refuses is stopped by
        the platform before the gate ever runs, and the envelope carries
        nothing at all — no answer, no `result`, no traceback."""
        akia = "AKIA" + "EFGH1234IJKL5678"
        status_code, body, raw = _post_invoke(
            {
                "input": _CALLER_QUESTION,
                "session_id": "pb-invoke-e2e",
                "input_context": {
                    "policy_passages": [dict(_CALLER_PASSAGES[0], text=_CALLER_PASSAGES[0]["text"] + f" Key {akia}.")]
                },
            }
        )
        assert status_code == 200
        assert body["status"] == "error", body
        assert not (body.get("output") or "")
        assert body["result"] is None
        assert body["answer"] is None
        assert akia not in raw
        assert "Traceback" not in raw
