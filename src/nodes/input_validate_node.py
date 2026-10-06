"""AgentCore Platform v1.0"""

# GOV-C2-001 — InputValidateNode
# Inner domain node 1 (first): domain-level validation of the policy question
# AND of the caller-supplied data contract (input_context).
#
# Distinct from PreProcessNode (trust + structural emptiness/length check):
# this node applies domain-business rules — minimum meaningful length,
# control-character stripping — and produces the canonical `query` string that
# the retrieval pipeline consumes.
#
# Caller-data contract (input_context — every field hostile until proven
# bounded; violations fail CLOSED with an error naming the FIELD, never
# echoing the value):
#
#   input_context = {
#     "channel": str matching ^[a-z0-9_]{1,32}$        (optional; consumed by
#                                                        the outer PreProcessNode)
#     "policy_passages": [                              (optional, 1..40 entries)
#       {
#         "id":       str ^[a-z0-9_]{1,32}$, unique     (required; rendered in
#                                                        citations — inert)
#         "source":   str, 1..160 chars, single line    (optional; rendered in
#                                                        citations — bounded,
#                                                        control-chars stripped,
#                                                        credential-scanned)
#         "text":     str, 20..4000 chars               (required; the passage
#                                                        body — whitespace
#                                                        collapsed, control
#                                                        chars stripped,
#                                                        credential-scanned)
#         "keywords": [str ^[a-z0-9]{2,24}$, ...] ≤15   (optional; scoring only)
#       }, ...
#     ]
#     "retrieval": {                                    (optional)
#       "top_k":           int 1..10                    (finite, bool rejected)
#       "score_threshold": number 0.78..1.0             (finite; may only
#                                                        TIGHTEN the configured
#                                                        grounding bar)
#     }
#   }
#
# When policy_passages validate, retrieval runs over the caller corpus instead
# of the built-in sample corpus; absent caller data degrades to the built-in
# baseline. Unknown input_context keys are ignored (no code path consumes
# them). Passage text/source are rendered into the answer document, so both
# are scanned at ingest with the same credential patterns the output gate
# enforces — a credential-bearing passage is rejected here, fail-closed,
# before it can ground an answer.
#
# Inner node — ANONYMOUS trust (the outer PreProcessNode with VERIFIED_EXTERNAL
# already enforced trust; inner nodes must be ANONYMOUS so the outer
# InvocationContext passes through the GraphNode boundary without rejection).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, NOT_OBJECT, TOO_SHORT

from src.nodes.post_process_node import _security_gate_output
from src.schemas.state import finite_in_range, to_json

logger = logging.getLogger(__name__)

# Minimum number of visible characters for a meaningful policy question.
_MIN_QUERY_CHARS = 3
# Strip ASCII control characters (except normal whitespace) that could carry
# injection payloads or corrupt downstream logging.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ── Caller-data validation bounds ────────────────────────────────────────────
_MAX_PASSAGES = 40
_MIN_TEXT_CHARS = 20
_MAX_TEXT_CHARS = 4000
_MAX_SOURCE_CHARS = 160
_MAX_KEYWORDS = 15
_ID_RE = re.compile(r"^[a-z0-9_]{1,32}$")
_KEYWORD_RE = re.compile(r"^[a-z0-9]{2,24}$")
_TOP_K_MIN, _TOP_K_MAX = 1, 10
# The configured grounding bar (config/config.yaml retrieval.score_threshold)
# is 0.78; a caller override may only TIGHTEN it, never loosen it.
_THRESHOLD_MIN, _THRESHOLD_MAX = 0.78, 1.0
# Rendered-source fallback for caller passages that omit a source label.
_CALLER_SOURCE_FALLBACK = "(caller-supplied source)"


def _clean_inline(raw: str) -> str:
    """Strip control chars and collapse all whitespace runs to single spaces.

    Applied to every caller string that renders into the answer document, so
    caller content cannot fake document structure (section separators, source
    lists) with embedded newlines, nor carry control characters.
    """
    return " ".join(_CONTROL_CHARS_RE.sub("", raw).split())


def _validate_passages(raw: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Validate and normalise caller-supplied policy passages.

    Returns (normalised_passages, error_message). Error messages name the
    field/index only — never the offending value. Unknown entry keys are
    dropped from the normalised output.
    """
    if not isinstance(raw, list):
        return None, "policy_passages must be a list"
    if not 1 <= len(raw) <= _MAX_PASSAGES:
        return None, f"policy_passages must contain between 1 and {_MAX_PASSAGES} entries"
    normalised: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return None, f"policy_passages[{i}] must be an object"

        pid = entry.get("id")
        if not isinstance(pid, str) or not _ID_RE.match(pid):
            return None, (f"policy_passages[{i}].id must be a string matching [a-z0-9_]{{1,32}}")
        if pid in seen_ids:
            return None, f"policy_passages[{i}].id duplicates an earlier passage id"
        seen_ids.add(pid)

        text = entry.get("text")
        if not isinstance(text, str):
            return None, f"policy_passages[{i}].text must be a string"
        text = _clean_inline(text)
        if not _MIN_TEXT_CHARS <= len(text) <= _MAX_TEXT_CHARS:
            return None, (
                f"policy_passages[{i}].text must be between {_MIN_TEXT_CHARS} "
                f"and {_MAX_TEXT_CHARS} characters after normalisation"
            )
        if _security_gate_output(text) is not None:
            return None, (f"policy_passages[{i}].text contains a disallowed credential-like pattern")

        source = entry.get("source", None)
        if source is None:
            source = _CALLER_SOURCE_FALLBACK
        else:
            if not isinstance(source, str):
                return None, f"policy_passages[{i}].source must be a string"
            source = _clean_inline(source)
            if not 1 <= len(source) <= _MAX_SOURCE_CHARS:
                return None, (
                    f"policy_passages[{i}].source must be between 1 and "
                    f"{_MAX_SOURCE_CHARS} characters after normalisation"
                )
            if _security_gate_output(source) is not None:
                return None, (f"policy_passages[{i}].source contains a disallowed credential-like pattern")

        keywords_raw = entry.get("keywords", [])
        if keywords_raw is None:
            keywords_raw = []
        if not isinstance(keywords_raw, list) or len(keywords_raw) > _MAX_KEYWORDS:
            return None, (f"policy_passages[{i}].keywords must be a list of at most {_MAX_KEYWORDS} entries")
        keywords: List[str] = []
        for j, kw in enumerate(keywords_raw):
            if not isinstance(kw, str) or not _KEYWORD_RE.match(kw):
                return None, (f"policy_passages[{i}].keywords[{j}] must be a string matching [a-z0-9]{{2,24}}")
            keywords.append(kw)

        normalised.append({"id": pid, "source": source, "text": text, "keywords": keywords})
    return normalised, None


def _validate_retrieval(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate caller retrieval overrides (top_k / score_threshold).

    Both numerics go through explicit finite+bounded checks and fail CLOSED:
    bools, strings, NaN, ±Infinity, and out-of-range magnitudes are all
    rejected with an error naming the field.
    """
    if not isinstance(raw, dict):
        return None, "retrieval must be an object"
    overrides: Dict[str, Any] = {}

    if "top_k" in raw:
        top_k = raw["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not _TOP_K_MIN <= top_k <= _TOP_K_MAX:
            return None, f"retrieval.top_k must be an integer between {_TOP_K_MIN} and {_TOP_K_MAX}"
        overrides["top_k"] = top_k

    if "score_threshold" in raw:
        threshold = raw["score_threshold"]
        parsed = finite_in_range(threshold, _THRESHOLD_MIN, _THRESHOLD_MAX)
        if parsed is None:
            return None, (
                f"retrieval.score_threshold must be a finite number between "
                f"{_THRESHOLD_MIN} and {_THRESHOLD_MAX} (the configured grounding "
                f"bar may only be tightened)"
            )
        overrides["score_threshold"] = parsed

    return overrides, None


class InputValidateNode(FunctionNode):
    """Domain validation of the policy question and caller data for GOV-C2-001.

    Applies business-rule checks beyond the structural check in PreProcessNode
    (minimum meaningful length, control-character stripping) and validates the
    caller-data contract carried in input_context — see the module docstring
    for the full contract. Produces the canonical `query` consumed by
    RetrieveNode, plus `caller_corpus` / `retrieval_overrides` when the caller
    supplied them and they validated.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        validated_input: str  — normalised question from PreProcessNode
                                Falls back to user_input for unit-test convenience.
        input_context:   dict — caller-supplied data (seeded by
                                _extra_initial_state via the context bridge)

    Output state keys (partial dict):
        query:               str
        caller_corpus:       str  — JSON list of validated passages (only when supplied)
        retrieval_overrides: str  — JSON dict of validated overrides (only when supplied)
        status:              str
        error_log:           list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        emit_progress("Checking the request...")
        raw = state.get("validated_input") or state.get("user_input", "")

        if not isinstance(raw, str) or not raw.strip():
            logger.error("InputValidateNode: query is empty or missing")
            emit_trace_event(
                "input_validate_failed",
                {"reason": "empty_query"},
                state,
            )
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["InputValidateNode: query is empty or missing"],
            }

        # ── Strip control chars + collapse whitespace ─────────────────────────
        cleaned = _CONTROL_CHARS_RE.sub("", raw)
        query = " ".join(cleaned.split()).strip()

        if len(query) < _MIN_QUERY_CHARS:
            logger.error(
                "InputValidateNode: query too short (%d < %d chars)",
                len(query),
                _MIN_QUERY_CHARS,
            )
            emit_trace_event(
                "input_validate_failed",
                {"reason": "query_too_short", "length": len(query)},
                state,
            )
            emit_progress(TOO_SHORT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": [f"InputValidateNode: query shorter than {_MIN_QUERY_CHARS} chars"],
            }

        # ── Caller-data contract (input_context) ──────────────────────────────
        input_context = state.get("input_context") or {}
        if not isinstance(input_context, dict):
            logger.error("InputValidateNode: input_context is not an object")
            emit_trace_event(
                "input_validate_failed",
                {"reason": "input_context_not_object"},
                state,
            )
            emit_progress(NOT_OBJECT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": ["InputValidateNode: input_context must be an object"],
            }

        result: dict[str, Any] = {}

        if "policy_passages" in input_context:
            passages, err = _validate_passages(input_context["policy_passages"])
            if err:
                logger.error("InputValidateNode: caller corpus rejected — %s", err)
                emit_trace_event(
                    "input_validate_failed",
                    {"reason": "caller_corpus_rejected", "field_error": err},
                    state,
                )
                emit_progress(INPUT_REJECTED)
                return {
                    "status": AgentStatus.SUCCESS.value,
                    "error_code": "INVALID_REQUEST",
                    "error_log": [f"InputValidateNode: {err}"],
                }
            result["caller_corpus"] = to_json(passages)

        if "retrieval" in input_context:
            overrides, err = _validate_retrieval(input_context["retrieval"])
            if err:
                logger.error("InputValidateNode: retrieval overrides rejected — %s", err)
                emit_trace_event(
                    "input_validate_failed",
                    {"reason": "retrieval_overrides_rejected", "field_error": err},
                    state,
                )
                emit_progress(INPUT_REJECTED)
                return {
                    "status": AgentStatus.SUCCESS.value,
                    "error_code": "INVALID_REQUEST",
                    "error_log": [f"InputValidateNode: {err}"],
                }
            if overrides:
                result["retrieval_overrides"] = to_json(overrides)

        logger.info(
            "InputValidateNode: query validated (%d chars, caller_corpus=%s)",
            len(query),
            "yes" if "caller_corpus" in result else "no",
        )
        emit_trace_event(
            "input_validate_complete",
            {
                "query_length": len(query),
                "caller_passage_count": (
                    len(input_context.get("policy_passages", []))
                    if isinstance(input_context.get("policy_passages"), list)
                    else 0
                ),
                "has_retrieval_overrides": "retrieval_overrides" in result,
            },
            state,
        )

        result["query"] = query
        result["status"] = AgentStatus.SUCCESS.value
        return result
