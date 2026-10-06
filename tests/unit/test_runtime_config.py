# GOV-C2-001 — Unit tests: runtime-config plumbing (config/config.yaml)
#
# The manifest split leaves runtime parameters in config/config.yaml. These
# tests prove the declared values are LIVE — read by _runtime_config(),
# forwarded by PolicyQAGraphNode._parent_config(), and actually reaching the
# inner domain nodes through a full compiled outer-graph invoke — not dead
# configuration text. (A config reader silently returning {} makes every node
# fall back to its hard-coded default while all happy-path tests stay green;
# the end-to-end probe below flips an observable outcome via the config file
# so that regression cannot hide.)

import pytest

from framework.schemas.agent_status import AgentStatus

# Grounds on the built-in corpus at 5/6 = 0.8333 ("official" is the one query
# content-token the passage does not cover) — BETWEEN the shipped 0.78 bar and
# the probe's tightened 0.90 bar, so the configured threshold decides the
# grounded/decline outcome observably.
_PROBE_QUESTION = "What is the official filing deadline period for an administrative appeal?"


def _patch_emit(monkeypatch):
    for mod in (
        "pre_process_node",
        "input_validate_node",
        "retrieve_node",
        "rerank_filter_node",
        "generate_answer_node",
        "output_format_node",
        "post_process_node",
    ):
        monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)


def _invoke(question, config=None):
    # config= mirrors every real entry point: AgentRegistry, src/api/server.py and
    # cli.py each read config/config.yaml themselves and hand the result to
    # Graph(config=...). The graph never opens the file itself.
    from framework.schemas.invocation_context import InvocationContext, TrustLevel
    from src.graph.graph import Graph

    agent = Graph(config=config or {})
    agent.compile()
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    return agent.invoke(question, ctx=ctx)


class TestRuntimeConfigFile:
    def test_shipped_config_parses_and_declares_expected_keys(self):
        from src.graph.graph import _runtime_config

        cfg = _runtime_config()
        assert cfg.get("max_retry") == 3
        assert cfg.get("timeout_s") == 30
        assert cfg["retrieval"]["top_k"] == 5
        assert cfg["retrieval"]["score_threshold"] == 0.78
        assert cfg["llm"]["temperature"] == 0.0
        assert cfg["security"]["s3_gate_enabled"] is True

    def test_parent_config_forwards_declared_settings(self):
        from src.graph.graph import PolicyQAGraphNode

        from src.graph.graph import _runtime_config

        configurable = PolicyQAGraphNode(runtime_config=_runtime_config())._parent_config()["configurable"]
        assert configurable["top_k"] == 5
        assert configurable["score_threshold"] == 0.78
        assert configurable["system_prompt_template"] == "prompts/gov_qa.j2"
        assert configurable["max_tokens"] == 3000

    def test_missing_or_malformed_file_degrades_to_empty(self, tmp_path, monkeypatch):
        import src.graph.graph as graph_mod

        monkeypatch.setattr(graph_mod, "_RUNTIME_CONFIG_PATH", tmp_path / "absent.yaml")
        assert graph_mod._runtime_config() == {}

        bad = tmp_path / "bad.yaml"
        bad.write_text("not: [valid yaml", encoding="utf-8")
        monkeypatch.setattr(graph_mod, "_RUNTIME_CONFIG_PATH", bad)
        assert graph_mod._runtime_config() == {}


class TestConfigReachesInnerGraphEndToEnd:
    """The declared retrieval settings must decide real outcomes end-to-end."""

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        _patch_emit(monkeypatch)

    def test_probe_question_scores_between_bars(self):
        # Guard the probe's meaningfulness: the probe question must score
        # strictly between the shipped bar (0.78) and the tightened bar (0.90)
        # on its best passage.
        from src.nodes.retrieve_node import RetrieveNode
        from src.schemas.state import from_json

        docs = from_json(RetrieveNode().execute({"query": _PROBE_QUESTION})["retrieved_docs"])
        top = docs[0]
        assert top["id"] == "GOV-KB-002"
        assert 0.78 <= top["score"] < 0.90

    def test_shipped_threshold_grounds_the_probe_question(self):
        result = _invoke(_PROBE_QUESTION)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "Grounded: YES" in result["output"]

    def test_tightened_threshold_flips_the_outcome(self, tmp_path, monkeypatch):
        # Same question, same pipeline — only config/config.yaml differs. The
        # tightened bar must flip the outcome to a decline, proving the value
        # travels config file → _parent_config → constructor injection →
        # RerankFilterNode through the full nested invoke.
        import yaml

        probe = tmp_path / "config.yaml"
        probe.write_text("retrieval:\n  top_k: 5\n  score_threshold: 0.90\n", encoding="utf-8")
        result = _invoke(_PROBE_QUESTION, config=yaml.safe_load(probe.read_text(encoding="utf-8")))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "Grounded: NO" in result["output"]

    def test_config_top_k_reaches_retrieve_node(self, tmp_path, monkeypatch):
        import src.graph.graph as graph_mod
        from src.nodes.retrieve_node import RetrieveNode
        from src.schemas.state import from_json

        probe = tmp_path / "config.yaml"
        probe.write_text("retrieval:\n  top_k: 2\n", encoding="utf-8")
        import yaml

        runtime = yaml.safe_load(probe.read_text(encoding="utf-8"))
        config = graph_mod.PolicyQAGraphNode(runtime_config=runtime)._parent_config()
        node = RetrieveNode(config=config)
        result = node.execute({"query": _PROBE_QUESTION})
        assert result["retrieved_count"] == 2
        assert len(from_json(result["retrieved_docs"])) == 2

    def test_absent_config_still_serves_with_node_defaults(self, tmp_path, monkeypatch):
        import src.graph.graph as graph_mod

        monkeypatch.setattr(graph_mod, "_RUNTIME_CONFIG_PATH", tmp_path / "absent.yaml")
        result = _invoke(_PROBE_QUESTION)
        # Node default threshold (0.78) applies — the probe question grounds.
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "Grounded: YES" in result["output"]
