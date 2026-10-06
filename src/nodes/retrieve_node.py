"""AgentCore Platform v1.0"""

# GOV-C2-001 — RetrieveNode
# Inner domain node 2: retrieve top_k candidate passages from the government
# policy knowledge base.
#
# Deterministic keyword-overlap retrieval over the active corpus: the
# VALIDATED caller-supplied passages (state["caller_corpus"], written by
# InputValidateNode) when the caller provided them, else the small read-only
# in-module sample corpus. Production wires the real vector store
# (config retrieval.vector_store.collection = "gov_policy_and_regulation_kb")
# — the installed framework SDK ships no vector-store client.
#
# Inner node — ANONYMOUS trust (trust is enforced at the outer pre_process).
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import RETRIEVAL_FAILED

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# English stopword / function-word set — dropped from the relevance signal so
# the score reflects content-term overlap, not filler or grammatical words.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "for",
        "in",
        "on",
        "and",
        "or",
        "what",
        "which",
        "how",
        "do",
        "does",
        "did",
        "can",
        "i",
        "my",
        "with",
        "about",
        "when",
        "where",
        "who",
        "whom",
        "this",
        "that",
        "it",
        "be",
        "by",
        "against",
        "from",
        "into",
        "over",
        "under",
        "than",
        "then",
        "there",
        "will",
        "shall",
        "must",
        "may",
        "should",
        "would",
        "could",
        "been",
        "being",
        "have",
        "has",
        "had",
        "not",
        "any",
        "all",
        "such",
        "at",
        "as",
        "so",
        "if",
        "you",
        "your",
        "we",
        "our",
        "me",
        "please",
        "tell",
        "need",
        "want",
        "get",
    }
)


def _stem(token: str) -> str:
    """Very light suffix stemmer so variants collapse (filing/filed -> fil,
    months/month -> month, subsidies/subsidy -> subsidy). Deterministic and
    dependency-free — good enough for keyword-overlap retrieval."""
    for suf in ("ing", "ies", "ed", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            if suf == "ies":
                return token[:-3] + "y"
            if suf == "s" and token.endswith("ss"):
                return token  # keep "process", "address" intact
            return token[: -len(suf)]
    return token


# Read-only built-in sample corpus (baseline when the caller supplies no
# passages — replaced by the real vector store in production).
# Each entry: id, source, text, keywords.
_POLICY_KB: tuple[Dict[str, Any], ...] = (
    {
        "id": "GOV-KB-001",
        "source": "行政手続法 (Administrative Procedure Act) Art. 6",
        "text": (
            "Administrative agencies must establish and publish standard processing "
            "periods for applications. When an application is received, the agency "
            "must begin its review without delay and notify the applicant of the "
            "expected processing period."
        ),
        "keywords": [
            "application",
            "processing",
            "period",
            "deadline",
            "agency",
            "review",
            "administrative",
            "procedure",
            "notify",
            "applicant",
        ],
    },
    {
        "id": "GOV-KB-002",
        "source": "行政不服審査法 (Administrative Appeal Act) Art. 18",
        "text": (
            "A request for administrative review (appeal) must in principle be filed "
            "within three months from the day following the day on which the person "
            "became aware of the disposition. A request filed after the period is "
            "dismissed unless there is a justifiable reason for the delay."
        ),
        "keywords": [
            "administrative",
            "review",
            "appeal",
            "filed",
            "three",
            "months",
            "period",
            "deadline",
            "disposition",
            "request",
            "dismissed",
        ],
    },
    {
        "id": "GOV-KB-003",
        "source": "マイナンバー法 (My Number Act) Art. 19",
        "text": (
            "The provision of Individual Numbers (My Number) to third parties is "
            "restricted to the cases specifically enumerated by law, such as social "
            "security, tax, and disaster-response procedures. Handling personal "
            "numbers outside these purposes is prohibited."
        ),
        "keywords": [
            "my",
            "number",
            "individual",
            "personal",
            "provision",
            "third",
            "party",
            "social",
            "security",
            "tax",
            "prohibited",
            "restricted",
        ],
    },
    {
        "id": "GOV-KB-004",
        "source": "デジタル庁 オンライン申請ガイドライン (Digital Agency e-Application Guideline)",
        "text": (
            "Citizens may submit administrative applications online through the "
            "Mynaportal service using My Number Card authentication. Online "
            "submissions receive an immediate acknowledgement number and can be "
            "tracked through the citizen's portal dashboard."
        ),
        "keywords": [
            "online",
            "application",
            "submit",
            "mynaportal",
            "digital",
            "agency",
            "authentication",
            "card",
            "acknowledgement",
            "citizen",
        ],
    },
    {
        "id": "GOV-KB-005",
        "source": "情報公開法 (Freedom of Information Act) Art. 5",
        "text": (
            "Any person may request disclosure of administrative documents held by an "
            "administrative organ. The organ must, in principle, make a disclosure "
            "decision within thirty days of receiving the request, though the period "
            "may be extended for justifiable reasons."
        ),
        "keywords": [
            "information",
            "disclosure",
            "request",
            "administrative",
            "documents",
            "thirty",
            "days",
            "period",
            "decision",
            "public",
        ],
    },
    {
        "id": "GOV-KB-006",
        "source": "補助金適正化法 (Subsidy Control Act) Art. 7",
        "text": (
            "Applicants for national subsidies must submit an application describing "
            "the purpose and expected cost. The granting agency examines eligibility "
            "and issues a grant decision; recipients must report on the use of funds "
            "and return any surplus."
        ),
        "keywords": [
            "subsidy",
            "subsidies",
            "grant",
            "application",
            "eligibility",
            "agency",
            "decision",
            "funds",
            "report",
            "cost",
            "national",
        ],
    },
)


def _tokens(text: str) -> List[str]:
    """Lowercase, stemmed content tokens (length > 1, non-stopword)."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _relevance(query_tokens: set[str], doc: Dict[str, Any]) -> float:
    """Fraction of query content-tokens covered by the doc text + keywords.

    Returns a value in [0.0, 1.0].  A passage that contains all of the query's
    content terms scores 1.0; an unrelated passage scores near 0.0.
    """
    if not query_tokens:
        return 0.0
    doc_tokens = set(_tokens(doc.get("text", "")))
    doc_tokens.update(_stem(k.lower()) for k in doc.get("keywords", []))
    overlap = query_tokens & doc_tokens
    return round(len(overlap) / len(query_tokens), 4)


class RetrieveNode(FunctionNode):
    """Retrieve top_k candidate policy passages for the query.

    Deterministic keyword-overlap retrieval over the active corpus — the
    validated caller corpus when present, else the built-in sample corpus.
    Production: real vector-store lookup on the configured collection.

    Inner node — ANONYMOUS trust (see module docstring).

    Configuration is injected through the constructor:
    DomainWorkflowGraph.register_nodes() passes the graph's config dict (the
    ``{"configurable": {...}}`` shape built by
    PolicyQAGraphNode._parent_config() from config/config.yaml). A validated
    caller ``retrieval.top_k`` override (state["retrieval_overrides"]) takes
    precedence; both paths are re-bounded here as defense in depth.

    Input state keys:
        query:               str — validated policy question (from InputValidateNode)
        caller_corpus:       str — JSON list of validated caller passages (optional)
        retrieval_overrides: str — JSON dict of validated overrides (optional)

    Output state keys (partial dict):
        retrieved_docs:  str   — JSON-serialised candidate list
        retrieved_count: int
        status:          str
        error_log:       list[str]  (only on ERROR)
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

        emit_progress("Searching the knowledge base...")
        query = state.get("query") or state.get("validated_input") or ""

        if not isinstance(query, str) or not query.strip():
            logger.error("RetrieveNode: query missing in state")
            emit_trace_event(
                "retrieve_failed",
                {"reason": "missing_query"},
                state,
            )
            emit_progress(RETRIEVAL_FAILED)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["RetrieveNode: query missing in state"],
            }

        configurable = (self._config or {}).get("configurable", {})
        top_k_raw: Any = configurable.get("top_k", _DEFAULT_TOP_K)

        # A validated caller override (InputValidateNode) takes precedence over
        # the configured default.
        overrides = from_json(state.get("retrieval_overrides"), {}) or {}
        if "top_k" in overrides:
            top_k_raw = overrides["top_k"]

        # Defense in depth: whatever the source, top_k must be a sane bounded
        # integer — anything else falls back to the default.
        if isinstance(top_k_raw, bool) or not isinstance(top_k_raw, int) or not 1 <= top_k_raw <= 10:
            top_k = _DEFAULT_TOP_K
        else:
            top_k = top_k_raw

        # Active corpus: validated caller passages replace the built-in sample
        # corpus when present; absent caller data degrades to the baseline.
        caller_corpus = from_json(state.get("caller_corpus"), None)
        if isinstance(caller_corpus, list) and caller_corpus:
            corpus: tuple[Dict[str, Any], ...] = tuple(caller_corpus)
            corpus_source = "caller"
        else:
            corpus = _POLICY_KB
            corpus_source = "builtin"

        query_tokens = set(_tokens(query))

        # Score every corpus passage (read-only; build a fresh local list —
        # never mutate the module-level corpus).
        scored: List[Dict[str, Any]] = []
        for doc in corpus:
            score = _relevance(query_tokens, doc)
            scored.append(
                {
                    "id": doc.get("id", "?"),
                    "source": doc.get("source", "unknown source"),
                    "text": doc.get("text", ""),
                    "score": score,
                }
            )

        # Rank by score desc, then id for determinism; keep the best top_k.
        scored.sort(key=lambda d: (-d["score"], d["id"]))
        candidates = scored[:top_k]

        logger.info(
            "RetrieveNode: query_tokens=%d candidates=%d top_score=%.2f",
            len(query_tokens),
            len(candidates),
            candidates[0]["score"] if candidates else 0.0,
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "query_token_count": len(query_tokens),
                "candidate_count": len(candidates),
                "top_score": candidates[0]["score"] if candidates else 0.0,
                "top_k": top_k,
                "corpus_source": corpus_source,
            },
            state,
        )

        return {
            "retrieved_docs": to_json(candidates),
            "retrieved_count": len(candidates),
            "status": AgentStatus.SUCCESS.value,
        }
