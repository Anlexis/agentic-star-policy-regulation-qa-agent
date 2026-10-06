# GOV-C2-001 — Unit tests: the caller-data contract (input_context)
#
# Every caller-controlled field is hostile until proven bounded:
#   - policy_passages: entry caps, inert ids, bounded/normalised text and
#     source, keyword lockdown, ingest credential scan — fail CLOSED, and the
#     rejected VALUE is never echoed into error logs (the field name is).
#   - retrieval overrides: finite+bounded numerics. NaN/Infinity parse fine
#     via json/float but compare False — a non-finite threshold would silently
#     disable the grounding bar, so every numeric goes through the
#     finite_in_range parser and is rejected.
#   - channel: inert identifier only.
# The retrieval pipeline computes REAL outputs from the validated caller
# corpus (it replaces the built-in sample corpus); absent caller data degrades
# to the built-in baseline.

import math

import pytest

from framework.schemas.agent_status import AgentStatus
from src.schemas.state import finite_in_range, from_json, to_json

_QUESTION = "What is the filing deadline period for an administrative appeal?"


def _passage(pid="agency_kb_001", **over):
    entry = {
        "id": pid,
        "source": "Municipal Ordinance No. 12 Art. 4",
        "text": (
            "Bicycle parking permits are issued within fourteen days of a " "complete application to the ward office."
        ),
        "keywords": ["bicycle", "parking", "permit", "application"],
    }
    entry.update(over)
    return entry


def _ctx(**over):
    ctx = {"policy_passages": [_passage()]}
    ctx.update(over)
    return ctx


class _NodeHarness:
    node_module = ""

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr(f"src.nodes.{self.node_module}.emit_trace_event", lambda *a, **k: None)


class TestCallerCorpusValidation(_NodeHarness):
    node_module = "input_validate_node"

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def _run(self, input_context):
        return self.node.execute({"validated_input": _QUESTION, "input_context": input_context})

    # ── Accepted shapes ───────────────────────────────────────────────────────

    def test_valid_passages_accepted_and_normalised(self):
        result = self._run(_ctx())
        assert result["status"] == AgentStatus.SUCCESS.value
        corpus = from_json(result["caller_corpus"])
        assert len(corpus) == 1
        assert corpus[0]["id"] == "agency_kb_001"
        assert corpus[0]["source"].startswith("Municipal Ordinance")

    def test_absent_caller_data_degrades_to_baseline(self):
        result = self._run({})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "caller_corpus" not in result
        assert "retrieval_overrides" not in result

    def test_text_whitespace_and_control_chars_normalised(self):
        text = "Line one\nLine two\x00 with\ttabs   and   runs of spaces padded out."
        result = self._run(_ctx(policy_passages=[_passage(text=text)]))
        assert result["status"] == AgentStatus.SUCCESS.value
        cleaned = from_json(result["caller_corpus"])[0]["text"]
        assert "\n" not in cleaned and "\x00" not in cleaned and "  " not in cleaned

    def test_source_absent_uses_fallback_marker(self):
        p = _passage()
        del p["source"]
        result = self._run(_ctx(policy_passages=[p]))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["caller_corpus"])[0]["source"] == "(caller-supplied source)"

    def test_unknown_passage_keys_dropped(self):
        result = self._run(_ctx(policy_passages=[_passage(internal_note="drop me")]))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "internal_note" not in from_json(result["caller_corpus"])[0]

    def test_unknown_input_context_keys_ignored(self):
        result = self._run(_ctx(unrelated_key={"anything": 1}))
        assert result["status"] == AgentStatus.SUCCESS.value

    # ── Rejected shapes (fail closed, field named, value never echoed) ────────

    @pytest.mark.parametrize(
        "passages",
        [
            "not-a-list",
            [],
            [_passage()] * 41,
            [42],
        ],
    )
    def test_structural_violations_rejected(self, passages):
        result = self._run({"policy_passages": passages})
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("policy_passages" in e for e in result["error_log"])

    @pytest.mark.parametrize(
        "bad_id",
        ["UPPER_CASE", "has space", "x" * 33, "", 123, None, "hyphen-id"],
    )
    def test_non_inert_id_rejected(self, bad_id):
        result = self._run(_ctx(policy_passages=[_passage(id=bad_id)]))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        joined = " ".join(result["error_log"])
        assert ".id" in joined
        if isinstance(bad_id, str) and bad_id:
            assert bad_id not in joined  # rejected value never echoed

    def test_duplicate_id_rejected(self):
        result = self._run(_ctx(policy_passages=[_passage(), _passage()]))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("duplicates" in e for e in result["error_log"])

    @pytest.mark.parametrize(
        "bad_text",
        [None, 42, "too short", "x" * 4001],
    )
    def test_text_bounds_rejected(self, bad_text):
        result = self._run(_ctx(policy_passages=[_passage(text=bad_text)]))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any(".text" in e for e in result["error_log"])

    def test_credential_bearing_text_rejected_and_not_echoed(self):
        secret = "api_key = sk-abcdefghij0123456789ABCDEF"
        result = self._run(_ctx(policy_passages=[_passage(text=f"A passage that leaks {secret} inline.")]))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        joined = " ".join(result["error_log"])
        assert ".text" in joined and "credential" in joined
        assert secret not in joined

    def test_credential_bearing_source_rejected(self):
        result = self._run(_ctx(policy_passages=[_passage(source="token=verysecretvalue123")]))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any(".source" in e for e in result["error_log"])

    @pytest.mark.parametrize("bad_source", [42, "", "x" * 161])
    def test_source_bounds_rejected(self, bad_source):
        result = self._run(_ctx(policy_passages=[_passage(source=bad_source)]))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any(".source" in e for e in result["error_log"])

    @pytest.mark.parametrize(
        "bad_keywords",
        ["not-a-list", ["ok"] * 16, ["UPPER"], ["x"], ["x" * 25], [42]],
    )
    def test_keyword_lockdown(self, bad_keywords):
        result = self._run(_ctx(policy_passages=[_passage(keywords=bad_keywords)]))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any(".keywords" in e for e in result["error_log"])

    def test_non_dict_input_context_rejected(self):
        result = self._run(["not", "a", "dict"])
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("input_context" in e for e in result["error_log"])


class TestRetrievalOverridesValidation(_NodeHarness):
    node_module = "input_validate_node"

    def setup_method(self):
        from src.nodes.input_validate_node import InputValidateNode

        self.node = InputValidateNode()

    def _run(self, retrieval):
        return self.node.execute({"validated_input": _QUESTION, "input_context": {"retrieval": retrieval}})

    @pytest.mark.parametrize("top_k", [1, 5, 10])
    def test_top_k_in_range_accepted(self, top_k):
        result = self._run({"top_k": top_k})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["retrieval_overrides"])["top_k"] == top_k

    @pytest.mark.parametrize(
        "top_k",
        [True, False, 0, 11, -1, 5.0, "5", None, float("nan"), float("inf")],
    )
    def test_top_k_violations_rejected(self, top_k):
        result = self._run({"top_k": top_k})
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("retrieval.top_k" in e for e in result["error_log"])

    @pytest.mark.parametrize("threshold", [0.78, 0.9, 1.0, 1])
    def test_threshold_in_range_accepted(self, threshold):
        result = self._run({"score_threshold": threshold})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["retrieval_overrides"])["score_threshold"] == float(threshold)

    # The non-finite matrix: every representation a caller can smuggle a
    # non-finite or out-of-range number through — string forms, raw floats,
    # bools, and over-magnitude values — must be rejected (fail CLOSED).
    @pytest.mark.parametrize(
        "threshold",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            "0.9",
            float("nan"),
            float("inf"),
            float("-inf"),
            1e308,
            0.5,
            0.7799,
            1.0001,
            -0.9,
            True,
            False,
            None,
            [0.9],
        ],
    )
    def test_threshold_violations_rejected(self, threshold):
        result = self._run({"score_threshold": threshold})
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("retrieval.score_threshold" in e for e in result["error_log"])

    def test_non_dict_retrieval_rejected(self):
        result = self._run("not-a-dict")
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("retrieval" in e for e in result["error_log"])


class TestFiniteInRange:
    """The finite+bounded parser itself."""

    def test_accepts_finite_in_range(self):
        assert finite_in_range(0.9, 0.78, 1.0) == 0.9
        assert finite_in_range(1, 0.78, 1.0) == 1.0

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), "0.9", "NaN", True, False, None, [], {}],
    )
    def test_rejects_non_finite_and_non_numeric(self, value):
        assert finite_in_range(value, 0.0, 1.0) is None

    def test_rejects_out_of_range(self):
        assert finite_in_range(1.01, 0.0, 1.0) is None
        assert finite_in_range(-0.01, 0.0, 1.0) is None

    def test_nan_comparison_hazard_is_real(self):
        # Regression documentation: NaN comparisons are always False, so a
        # naive `score >= threshold` check with a NaN threshold silently
        # filters everything (or nothing, depending on polarity). The parser
        # exists to keep NaN out of every comparison.
        nan = float("nan")
        assert not nan >= 0.78 and not nan <= 0.78 and math.isnan(nan)


class TestChannelValidation(_NodeHarness):
    node_module = "pre_process_node"

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def _run(self, input_context):
        return self.node.execute({"user_input": _QUESTION, "input_context": input_context})

    def test_valid_channel_accepted(self):
        result = self._run({"channel": "web_portal"})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["enriched_context"])["channel"] == "web_portal"

    def test_absent_channel_defaults_unknown(self):
        result = self._run({})
        assert result["status"] == AgentStatus.SUCCESS.value
        assert from_json(result["enriched_context"])["channel"] == "unknown"

    @pytest.mark.parametrize("channel", ["WEB", "has space", "x" * 33, "", 42, None, "hyphen-ated"])
    def test_invalid_channel_rejected_and_not_echoed(self, channel):
        result = self._run({"channel": channel})
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        joined = " ".join(result["error_log"])
        assert "input_context.channel" in joined
        if isinstance(channel, str) and channel:
            assert channel not in joined

    def test_non_dict_input_context_rejected(self):
        result = self._run("not-a-dict")
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("input_context" in e for e in result["error_log"])


class TestRetrieveWithCallerCorpus(_NodeHarness):
    node_module = "retrieve_node"

    def setup_method(self):
        from src.nodes.retrieve_node import RetrieveNode

        self.node = RetrieveNode()

    def test_caller_corpus_replaces_builtin(self):
        corpus = [
            {
                "id": "agency_kb_001",
                "source": "Municipal Ordinance No. 12 Art. 4",
                "text": (
                    "Bicycle parking permits are issued within fourteen days of "
                    "a complete application to the ward office."
                ),
                "keywords": ["bicycle", "parking", "permit", "application"],
            }
        ]
        result = self.node.execute(
            {
                "query": "How long does a bicycle parking permit application take?",
                "caller_corpus": to_json(corpus),
            }
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        docs = from_json(result["retrieved_docs"])
        assert [d["id"] for d in docs] == ["agency_kb_001"]
        assert all(not d["id"].startswith("GOV-KB") for d in docs)

    def test_absent_caller_corpus_uses_builtin(self):
        result = self.node.execute({"query": _QUESTION})
        docs = from_json(result["retrieved_docs"])
        assert docs and docs[0]["id"].startswith("GOV-KB")

    def test_caller_top_k_override_honoured(self):
        result = self.node.execute({"query": _QUESTION, "retrieval_overrides": to_json({"top_k": 2})})
        assert result["retrieved_count"] == 2

    def test_malformed_stored_override_falls_back(self):
        # Defense in depth: the override is validated upstream; if a malformed
        # value ever reaches this node it falls back to the configured default
        # instead of being trusted.
        result = self.node.execute({"query": _QUESTION, "retrieval_overrides": to_json({"top_k": 9999})})
        assert result["retrieved_count"] == 5


class TestRerankTightenOnly(_NodeHarness):
    node_module = "rerank_filter_node"

    def setup_method(self):
        from src.nodes.rerank_filter_node import RerankFilterNode

        self.node = RerankFilterNode()

    def _docs(self, score):
        return to_json([{"id": "agency_kb_001", "source": "Ordinance", "text": "t" * 30, "score": score}])

    def test_caller_threshold_tightens_the_bar(self):
        # 0.80 clears the default 0.78 bar but not a caller-tightened 0.90 bar.
        result = self.node.execute(
            {
                "retrieved_docs": self._docs(0.80),
                "retrieval_overrides": to_json({"score_threshold": 0.90}),
            }
        )
        assert result["grounded"] is False

    def test_caller_threshold_cannot_loosen_the_bar(self):
        # A (hypothetical) loose override never lowers the effective bar: the
        # node takes max(configured, caller) and floors the result.
        result = self.node.execute(
            {
                "retrieved_docs": self._docs(0.60),
                "retrieval_overrides": to_json({"score_threshold": 0.10}),
            }
        )
        assert result["grounded"] is False

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", float("nan"), float("inf"), None, True])
    def test_non_finite_stored_override_never_loosens(self, bad):
        # Defense in depth: a non-finite value reaching this node is ignored;
        # the configured bar stays in force (0.60 stays excluded).
        # json round-trips bare NaN/Infinity, so the stored JSON really can
        # carry a non-finite number into this node.
        result = self.node.execute(
            {
                "retrieved_docs": self._docs(0.60),
                "retrieval_overrides": to_json({"score_threshold": bad}),
            }
        )
        assert result["grounded"] is False
