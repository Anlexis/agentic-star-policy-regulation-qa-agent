"""AgentCore Platform v1.0"""

# GOV-C2-001 — PreProcessNode
# Outer backbone pre_process slot (trust gate + question input validation).
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL trust (required_trust_level — the framework
#     denies lower-trust callers before execute() runs)
#   - Reject empty / non-string / over-long input early (fail-fast)
#   - Bound the caller channel label to an inert identifier (fail-closed)
#   - Normalise the policy question (strip, collapse whitespace)
#   - Write validated_input + enriched_context to State
#   - Emit an audit event for every validation decision
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INVALID_VALUE, NOT_OBJECT, TOO_LONG

from src.schemas.state import to_json

logger = logging.getLogger(__name__)

# Upper bound on a single policy question (chars). Guards against oversized /
# abusive input reaching the retrieval pipeline.
_MAX_QUESTION_CHARS = 2000
_WHITESPACE_RE = re.compile(r"\s+")
# The caller channel label is rendered into internal metadata only, but it is
# caller-controlled — lock it to an inert identifier (fail-closed on anything
# else).
_CHANNEL_RE = re.compile(r"^[a-z0-9_]{1,32}$")


class PreProcessNode(FunctionNode):
    """Input validation for GOV-C2-001.

    Validates the caller-supplied policy question before the RAG workflow
    runs.  This is the outer backbone's pre_process slot — the only node with
    VERIFIED_EXTERNAL trust so that unauthenticated or anonymous callers are
    rejected here (fail-fast; inner domain nodes carry ANONYMOUS trust and
    never see untrusted input directly).

    Input state keys:
        user_input:    str  — caller-supplied natural-language policy question
        input_context: dict — caller-supplied request metadata / data contract

    Output state keys (partial dict):
        validated_input:  str        — normalised question string
        enriched_context: str        — JSON-serialised channel metadata
        status:           str        — AgentStatus.SUCCESS.value or ERROR
        error_log:        list[str]  — set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        emit_progress("Checking the request...")
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {}) or {}

        # ── input_context shape + channel label (caller-controlled) ───────────
        if not isinstance(input_context, dict):
            logger.warning("PreProcessNode: input_context is not an object")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "input_context_not_object"},
                state,
            )
            emit_progress(NOT_OBJECT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": ["PreProcessNode: input_context must be an object"],
            }
        channel = input_context.get("channel", "unknown")
        if channel != "unknown" and (not isinstance(channel, str) or not _CHANNEL_RE.match(channel)):
            # Fail closed; name the field, never echo the rejected value.
            logger.warning("PreProcessNode: channel label rejected")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "invalid_channel_label"},
                state,
            )
            emit_progress(INVALID_VALUE)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "INVALID_REQUEST",
                "error_log": ["PreProcessNode: input_context.channel must be a string " "matching [a-z0-9_]{1,32}"],
            }

        # ── Emptiness / type check ────────────────────────────────────────────
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            logger.warning("PreProcessNode: user_input is empty or missing")
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "empty_input"},
                state,
            )
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        # ── Normalise (strip + collapse internal whitespace) ──────────────────
        normalised = _WHITESPACE_RE.sub(" ", user_input.strip())

        # ── Length bound ──────────────────────────────────────────────────────
        if len(normalised) > _MAX_QUESTION_CHARS:
            logger.warning(
                "PreProcessNode: question too long (%d > %d chars)",
                len(normalised),
                _MAX_QUESTION_CHARS,
            )
            emit_trace_event(
                "pre_process_validation_failed",
                {"reason": "question_too_long", "length": len(normalised)},
                state,
            )
            emit_progress(TOO_LONG)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "QUESTION_TOO_LONG",
                "error_log": [f"PreProcessNode: question exceeds {_MAX_QUESTION_CHARS} chars"],
            }

        # ── Success ───────────────────────────────────────────────────────────
        logger.info("PreProcessNode: validated question (%d chars)", len(normalised))
        emit_trace_event(
            "pre_process_validated",
            {"question_length": len(normalised)},
            state,
        )

        return {
            "validated_input": normalised,
            "enriched_context": to_json(
                {
                    "source": "PolicyRegulationQAAgent",
                    "channel": channel,
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }
