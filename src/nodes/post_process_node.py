"""AgentCore Platform v1.0"""

# GOV-C2-001 — PostProcessNode
# Outer backbone post_process slot: output gate + expose the final policy
# answer as formatted_output and result.
#
# Output-gate responsibility (two independent layers, each with its own audit
# event):
#   (1) verbatim internal-state embedding — raw JSON blobs held in State
#       (retrieved candidates, the validated caller corpus, retrieval
#       overrides, channel metadata) must never appear verbatim in the
#       caller-facing document; any embedded copy is replaced with [REDACTED];
#   (2) credential scan — the answer document is scanned for credential-like
#       patterns before being returned to the caller. Recognition is the
#       FRAMEWORK's own detect_credentials(), widened by a few domain-extra
#       patterns; it is never narrower. The framework scans every node result
#       with that same detector and RAISES on a hit, and a raise makes
#       BaseNode.__call__ discard this node's whole return — including the
#       clearing below. A shape the framework catches and this gate missed is
#       therefore a containment BYPASS, not merely a narrower gate: the two
#       cannot be allowed to drift apart. (InputValidateNode imports this same
#       helper for its caller-corpus ingest check, so ingest widens with it.)
#
# Blocking CONTAINS, it does not merely flag. AgentBaseGraph resolves the
# caller-facing envelope value as `formatted_output or result` WITHOUT
# consulting status, so returning ERROR while leaving the output-bearing state
# populated still ships the refused answer inside the error envelope. On a
# violation this node therefore overwrites every field that carries answer text
# or a payload (see _CONTENT_FIELDS), and the formatted_output replacement is
# deliberately NON-EMPTY — a falsy stand-in ("" or {}) does not suppress that
# `or result` fallback, it ACTIVATES it. Violations name the pattern TYPE only;
# the offending value is never echoed, because echoing it would put the refused
# string back into this node's own result where the framework scan raises and
# discards the clearing.
#
# The domain output gate is a module-level function (_security_gate_output)
# called from inside execute() — NOT an instance method on the node class
# (an instance method of that name would be auto-wrapped by the framework and
# fail on the real invoke path).
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, OUTPUT_BLOCKED, TOO_LONG

logger = logging.getLogger(__name__)

# Domain-EXTRA credential patterns, applied ON TOP OF the framework's
# detect_credentials(). These only ever WIDEN recognition (looser bounds than
# the framework's, plus a `key = value` assignment shape the framework does not
# model); the gate must never be narrower than the detector the framework
# itself enforces one layer up — see the module docstring.
_DOMAIN_EXTRA_PATTERNS: List[tuple[str, str]] = [
    (r"(?:sk|pk|ak)-[A-Za-z0-9]{16,}", "api_key_pattern"),
    (r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "jwt_pattern"),
    (r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", "bearer_token"),
    (
        r"(?:password|passwd|secret|api_key|token|access_key|private_key)" r"\s*[:=]\s*\S{8,}",
        "credential_assignment",
    ),
]

# State fields whose raw (JSON) string values must never be embedded verbatim
# in the caller-facing document. The document is assembled from answer +
# citations only; a verbatim copy of any of these means internal pipeline
# state leaked into the external surface.
_BLOCKED_STATE_FIELDS = frozenset(
    {
        "retrieved_docs",
        "caller_corpus",
        "retrieval_overrides",
        "enriched_context",
    }
)


def _security_gate_output(content: str) -> Optional[str]:
    """Scan output for disallowed credential/secret patterns.

    Returns the violation TYPE name, or None if the output is clean. The name
    only — the matched value is never returned, so a caller putting this into
    error_log cannot re-introduce the refused string into a node result (where
    the framework credential scan would raise and discard the whole return).

    Recognition is the FRAMEWORK's detect_credentials() first, then the
    domain-extra patterns: the union is a strict SUPERSET of what the framework
    refuses. Anything narrower would be a containment bypass rather than a
    lenient gate (see the module docstring).

    Module-level function (not a node instance method) — this form keeps the
    gate outside the framework's node-method auto-wrapping.
    """
    findings = detect_credentials(content)
    if findings:
        return str(findings[0]["type"])
    for pattern, name in _DOMAIN_EXTRA_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return name
    return None


# Outer-state fields that carry answer text or a payload. A blocked response
# must overwrite EVERY one of them — the envelope reads state, not this node's
# intent. Keyed one-to-one to PolicyQAGraphNode.merge_output()'s delta (plus
# `result`, which this node owns); tests/unit/test_nodes.py pins the two
# together so a future domain field cannot quietly join the state and be
# released on a refused response.
_CONTENT_FIELDS: dict[str, Any] = {
    "answer": None,
    "answer_document": None,
    "citations": None,
    "filtered_docs": None,
    "result": None,
}

# Inert provenance carried by merge_output: a bare boolean, no content. Pinned
# content-free rather than left at its pre-gate value, so a refusal can never
# be read back as a grounded success.
# `error_code` is a fixed internal reason code (EMPTY_INPUT / INVALID_REQUEST /
# ...), chosen by this template and never assembled from request data, so it
# carries nothing a containment sweep would need to withhold. Pinned empty here
# for the same reason as `grounded`: a refusal must not be readable back as a
# completed decline.
_INERT_FIELDS: dict[str, Any] = {"grounded": False, "error_code": ""}


def _withheld_state(notice: str) -> dict[str, Any]:
    """The state delta that CONTAINS a blocked response.

    `formatted_output` is deliberately non-empty: AgentBaseGraph.get_output()
    resolves the caller-facing value as `formatted_output or result`, so a
    falsy stand-in would fall through to the pre-gate value and ship exactly
    what the gate just refused.
    """
    return {"formatted_output": notice, **_CONTENT_FIELDS, **_INERT_FIELDS}


def _redact_blocked_fields(content: str, state: AgentState) -> tuple[str, List[str]]:
    """Replace verbatim embeddings of blocked internal-state values.

    Returns (sanitised_content, redacted_field_names). Only string values
    longer than 10 chars are considered — short scalars cannot meaningfully
    identify an internal blob and would over-redact.
    """
    redacted: List[str] = []
    for field in sorted(_BLOCKED_STATE_FIELDS):
        val = state.get(field)
        if isinstance(val, str) and len(val) > 10 and val in content:
            content = content.replace(val, "[REDACTED]")
            redacted.append(field)
    return content, redacted


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Apply the output gate and expose the final policy answer.

    Outer backbone post_process slot.  Declared ANONYMOUS — trust was
    already enforced at PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        answer_document: str   — formatted cited answer from inner OutputFormatNode

    Output state keys (partial dict):
        formatted_output: str
        result:           str        — the gated document on success; None on a block
        status:           str
        error_log:        list[str]  (only on ERROR)
        On a gate block additionally: answer / answer_document / citations /
        filtered_docs cleared and grounded pinned False (see _CONTENT_FIELDS).
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }
        answer_document: str = state.get("answer_document") or state.get("result") or ""

        # ── Fallback for empty answer ─────────────────────────────────────────
        if not answer_document.strip():
            logger.warning("PostProcessNode: answer_document is empty — using fallback message")
            answer_document = "[Policy Q&A] No answer content was generated. " "Check error_log for upstream failures."

        # ── Layer 1: verbatim internal-state embedding ────────────────────────
        answer_document, redacted_fields = _redact_blocked_fields(answer_document, state)
        if redacted_fields:
            logger.error(
                "PostProcessNode: internal state embedded verbatim in output — %s",
                ", ".join(redacted_fields),
            )
            emit_trace_event(
                "post_process_blocked_field_redaction",
                {"fields": redacted_fields},
                state,
            )

        # ── Layer 2: credential scan ──────────────────────────────────────────
        violation = _security_gate_output(answer_document)
        if violation:
            logger.error("PostProcessNode: credential pattern detected in output — %s", violation)
            emit_trace_event(
                "post_process_output_gate_violation",
                {"violation": violation},
                state,
            )
            sanitised = (
                f"[ANSWER REDACTED: output contained a disallowed pattern "
                f"({violation}). Contact the policy KB administrator.]"
            )
            # CONTAINMENT: blocking is not a status flip. Every output-bearing
            # field is overwritten here (see _withheld_state) because the
            # envelope reads state directly and its `formatted_output or
            # result` resolution ignores status. The violation names the
            # pattern TYPE only — never the matched value.
            emit_progress(OUTPUT_BLOCKED)
            return {
                **_withheld_state(sanitised),
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output gate detected a credential pattern — {violation}"],
            }

        logger.info("PostProcessNode: output gate passed — length=%d", len(answer_document))
        emit_trace_event(
            "post_process_complete",
            {"output_length": len(answer_document)},
            state,
        )

        return {
            "formatted_output": answer_document,
            "result": answer_document,
            "status": AgentStatus.SUCCESS.value,
        }
