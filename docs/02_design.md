# Template Design Specification — GOV-C2-001 Policy & Regulation Q&A Agent

## Position in AgentCore Architecture

| Aspect | Value |
|---|---|
| Agent Class | PolicyRegulationQAAgent |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Pattern | Cat 2 — RAGAgent (two-layer nested workflow) |

**Three-layer separation**:
- State: flat TypedDict composition (no Pydantic — msgpack incompatible); all
  dict/list-valued fields stored as JSON-serialised strings
- Node: framework base-class inheritance (Template Method:
  `execute(self, state: dict) -> dict` override only)
- Graph: composition (`register_nodes()` for node substitution; Cat 2 nested
  via `GraphNode`)

## Domain Context

Government policy and administrative-regulation Q&A for Japanese public-sector
staff and citizens. Retrieves from an official policy knowledge base
(Digital Agency / デジタル庁, METI / 経済産業省, e-Gov 法令 sources) and returns a
grounded, cited answer about regulations, procedures, and policy guidelines.

**Risk posture**: government guidance is life/safety-adjacent — an incorrect or
ungrounded answer can mislead a citizen about a legal obligation or benefit.
The template therefore runs deterministic, fully grounded generation
(`temperature: 0.0`) and a raised retrieval bar (`score_threshold: 0.78`); when
no passage clears the bar the agent declines rather than guesses.

## Architecture Overview

### Backbone (outer AgentBaseGraph — fixed 5-node pipeline)

```
START → initialize → pre_process → main(GraphNode) → post_process → finalize → END
                                         ↓ (retry, max 3)
                                       pre_process
```

### Inner Domain Workflow (DomainWorkflowGraph — linear 5-node RAG pipeline)

```
START → input_validate → retrieve → rerank_filter → generate_answer
          → output_format → END
```

### Caller-data contract (input_context)

`POST /invoke` accepts an optional `input_context` object (serialized size
capped at 256 KB at the adapter) carrying per-request caller data. Every field
is hostile until proven bounded — validation fails CLOSED with an error naming
the field, and rejected values are never echoed:

| Field | Bounds | Consumer |
|---|---|---|
| `channel` | string `[a-z0-9_]{1,32}` | PreProcessNode (metadata only) |
| `policy_passages` | list, 1–40 entries | InputValidateNode → RetrieveNode |
| `policy_passages[].id` | string `[a-z0-9_]{1,32}`, unique | citations (inert identifier) |
| `policy_passages[].source` | string 1–160 chars, single line, credential-scanned | citations |
| `policy_passages[].text` | string 20–4000 chars, whitespace-normalised, control chars stripped, credential-scanned | retrieval + answer body |
| `policy_passages[].keywords` | ≤15 × `[a-z0-9]{2,24}` | retrieval scoring |
| `retrieval.top_k` | int 1–10 | RetrieveNode |
| `retrieval.score_threshold` | finite number 0.78–1.0 (may only TIGHTEN the configured bar) | RerankFilterNode |

Validated caller passages REPLACE the built-in sample corpus for that request;
absent caller data degrades to the built-in baseline. Both numerics pass a
finite+bounded parser (`finite_in_range`) — NaN/Infinity arrive parseable via
JSON but compare False, which would silently disable the grounding bar, so
non-finite values are rejected outright.

**Boundary note (outer→inner)**: the framework's `GraphNode.execute()` does not
forward `input_context` on `subgraph.invoke()` (SDK 1.0.1). The template
bridges it: `PolicyQAGraphNode.extract_input()` stashes the outer state's
`input_context` in a ContextVar, and `DomainWorkflowGraph._extra_initial_state()`
seeds it into the inner state (`src/graph/context_bridge.py`).


**Completion is not the same as answering.** A run that ends with
`AgentStatus.SUCCESS` reports that the request was handled safely to a defined
end, not that the request was carried out. A value the caller can correct (an
out-of-contract field, an empty or over-long request) ends this way so the
caller receives the reason and can send a corrected request on the same
conversation; terminating instead would end the calling surface's turn and
surface only an exception type, leaving the reason reachable solely from the
audit trail. The reason travels as `error_code` in State, every later domain
node passes through without doing work once it is set — the reason settled
first is the one the caller receives, never a second vaguer one from a later
node — the structured output fields are withheld, and `PostProcessNode` renders the
reason as a static caller-facing sentence.

Two classes keep terminating, and must not be folded into the above: content
the agent refuses outright (an instruction-override payload — re-sending a
reworded variant is not a correction), and a breach of a contract the caller
cannot influence.

### Node Configuration

| Node | Class | File | Trust | Responsibility | Input Keys | Output Keys |
|------|-------|------|-------|---------------|------------|-------------|
| initialize | InitializeNode | framework | — | session init | — | session_id, schema_version |
| pre_process | PreProcessNode | src/nodes/pre_process_node.py | VERIFIED_EXTERNAL | trust gate + question validation + channel lockdown | user_input, input_context | validated_input, enriched_context |
| main | PolicyQAGraphNode | src/graph/graph.py | — | delegates to DomainWorkflowGraph; bridges input_context | validated_input, input_context | answer, filtered_docs, citations, grounded, answer_document |
| post_process | PostProcessNode | src/nodes/post_process_node.py | ANONYMOUS | output gate (internal-state redaction + credential scan) + set formatted_output; on a block, CLEAR every output-bearing field | answer_document, answer, citations, filtered_docs | formatted_output, result (+ answer / answer_document / citations / filtered_docs / grounded on a block) |
| finalize | FinalizeNode | framework | — | response metadata | — | response_metadata, total_time_ms |
| input_validate (inner) | InputValidateNode | src/nodes/input_validate_node.py | ANONYMOUS | domain question validation + caller-data contract validation | validated_input, input_context | query, caller_corpus, retrieval_overrides |
| retrieve (inner) | RetrieveNode | src/nodes/retrieve_node.py | ANONYMOUS | top_k passage retrieval over the active corpus (caller or built-in) | query, caller_corpus, retrieval_overrides | retrieved_docs, retrieved_count |
| rerank_filter (inner) | RerankFilterNode | src/nodes/rerank_filter_node.py | ANONYMOUS | rerank + score_threshold filter (tighten-only overrides, hard floor 0.75) | retrieved_docs, retrieval_overrides | filtered_docs, grounded |
| generate_answer (inner) | GenerateAnswerNode | src/nodes/generate_answer_node.py | ANONYMOUS | grounded answer + citations (deterministic extractive synthesis) | filtered_docs, query | answer, citations |
| output_format (inner) | OutputFormatNode | src/nodes/output_format_node.py | ANONYMOUS | assemble final cited answer document | answer, citations | answer_document, result |

### Data Flow

```
user_input (natural-language policy question) + input_context (caller data)
    │
    ▼ PreProcessNode (VERIFIED_EXTERNAL)
validated_input (normalised question)
enriched_context (JSON string)
    │
    ▼ PolicyQAGraphNode → context bridge → DomainWorkflowGraph
    │   InputValidateNode  → query, caller_corpus, retrieval_overrides
    │   RetrieveNode       → retrieved_docs (JSON list), retrieved_count
    │   RerankFilterNode   → filtered_docs (JSON list), grounded (bool)
    │   GenerateAnswerNode → answer (grounded / decline), citations (JSON)
    │   OutputFormatNode   → answer_document (str), result (str)
    ▼ merge_output
answer, filtered_docs, citations, grounded, answer_document → outer state
    │
    ▼ PostProcessNode (ANONYMOUS, output gate)
formatted_output (gated answer_document), result
    │   on a BLOCK: formatted_output = withholding notice (truthy);
    │               result / answer / answer_document / citations /
    │               filtered_docs cleared; grounded pinned False
    ▼ PolicyRegulationQAAgent.get_output()
output = formatted_output (success) | formatted_output or None (non-success)
result surfaced ONLY when status == SUCCESS
```

### State Definition

| Field | Type | Purpose | Producer |
|-------|------|---------|----------|
| validated_input | NotRequired[Optional[str]] | Normalised policy question | PreProcessNode |
| enriched_context | NotRequired[Optional[str]] | JSON: {source, channel} | PreProcessNode |
| query | NotRequired[Optional[str]] | Cleaned/normalised query | InputValidateNode |
| caller_corpus | NotRequired[Optional[str]] | JSON: validated caller passages | InputValidateNode |
| retrieval_overrides | NotRequired[Optional[str]] | JSON: validated {top_k, score_threshold} | InputValidateNode |
| retrieved_docs | NotRequired[Optional[str]] | JSON: [{id, source, text, score}] top_k | RetrieveNode |
| retrieved_count | NotRequired[Optional[int]] | Candidate passage count | RetrieveNode |
| filtered_docs | NotRequired[Optional[str]] | JSON: passages clearing score_threshold | RerankFilterNode |
| grounded | NotRequired[Optional[bool]] | ≥1 passage cleared threshold | RerankFilterNode |
| answer | NotRequired[Optional[str]] | Grounded answer or decline message | GenerateAnswerNode |
| citations | NotRequired[Optional[str]] | JSON: [{id, source}] | GenerateAnswerNode |
| answer_document | NotRequired[Optional[str]] | Final cited answer document | OutputFormatNode |
| result | NotRequired[Optional[str]] | Same as answer_document (backbone convention) | OutputFormatNode / PostProcessNode |

**Serialisation constraint**: all dict/list-valued fields use JSON-serialised
`Optional[str]` (LangGraph checkpoints use msgpack — raw dicts/Pydantic models
corrupt silently). `to_json()` / `from_json()` helpers are defined in
`src/schemas/state.py` and used at every producer/consumer boundary — one
contract end-to-end. `finite_in_range()` (same module) is the mandatory parser
for every caller-controlled numeric.

**Prohibited**: re-declaring `formatted_output` (inherited from AgentState),
credentials in State, Pydantic models.

### Retrieval / RAG Configuration (config/config.yaml)

Runtime parameters live in `config/config.yaml` (the static registration
manifest `config/agent.yaml` is flat and holds no runtime block):

```yaml
max_retry: 3
timeout_s: 30
llm:
  system_prompt_template: prompts/gov_qa.j2
  temperature: 0.0
  max_tokens: 3000
retrieval:
  vector_store:
    collection: gov_policy_and_regulation_kb
  top_k: 5
  score_threshold: 0.78   # raised bar — life/safety-adjacent guidance
  hybrid_search: false
security:
  s3_gate_enabled: true
```

- **top_k: 5** — retrieve the 5 best candidate passages.
- **score_threshold: 0.78** — only high-confidence passages ground an answer;
  below it the agent declines (grounding discipline). `RerankFilterNode`
  additionally floors any configured value at 0.75.
- **temperature: 0.0** — deterministic generation; no creative deviation from
  source policy text.

The values are live end-to-end: the entry point reads this file and passes it as
`Graph(config=...)`, `PolicyQAGraphNode._parent_config()` exposes it under
`config["configurable"]`, and `DomainWorkflowGraph.register_nodes()` injects it
into the config-consuming node constructors. `tests/unit/test_runtime_config.py`
proves the path by flipping the grounding outcome through a config change.

## Security Configuration

| Concern | Gate | Implementation |
|-------|------|---------------|
| Trust enforcement | pre_process | PreProcessNode `required_trust_level = VERIFIED_EXTERNAL` |
| Input validation | pre_process + input_validate | Structural question bounds; full caller-data contract (bounds table above): inert identifiers, entry/length caps, finite+bounded numerics, ingest credential scan — fail closed, rejected values never echoed |
| Output gate | post_process | Module-level `_security_gate_output()` credential scan + verbatim internal-state redaction (`_redact_blocked_fields`), each with its own audit event. Recognition delegates to the framework's own `detect_credentials()`, widened by domain-extra patterns — never narrower (see "Output boundary contract"). On a credential hit the node returns status=ERROR **and contains**: `formatted_output` becomes a non-empty withholding notice and `result`, `answer`, `answer_document`, `citations`, `filtered_docs` are cleared (`grounded` pinned False). Violations name the pattern TYPE only, never the matched value |
| Output envelope | `PolicyRegulationQAAgent.get_output()` | `result` surfaced only on the gated success path; on any non-success the base `formatted_output or result` fallback is re-resolved as `formatted_output or None` |
| Audit logging | every node | `emit_trace_event()` in every node's `execute()` (at least one domain-specific event) |
| Credential handling | — | No credentials in State; secrets via InvocationContext only |

### Output boundary contract

The caller-facing answer document renders ONLY: content from the validated
active corpus (built-in, or caller passages that passed ingest validation),
citations naming validated ids/sources, and the grounded flag. Quoted
regulatory values (deadlines, amounts, article numbers) are reproduced exactly
as written in the source passage — accuracy is the domain requirement, so the
gate never rewrites statutory content; it blocks credential-bearing output and
verbatim internal-state embeddings instead.

**Blocking contains, it does not merely flag.** The framework's
`AgentBaseGraph.get_output()` resolves the caller-facing value as
`formatted_output or result` **without consulting status**, and
`PolicyQAGraphNode.merge_output()` has already copied the pre-gate `answer`,
`citations`, `filtered_docs` and `answer_document` into outer state. Three
consequences shape the contract:

1. A gate that only flips the status still ships the refused answer inside the
   error envelope. `PostProcessNode` therefore overwrites every field carrying
   answer text or a payload (`_CONTENT_FIELDS`), and the inventory is pinned
   against `merge_output()`'s key set by a test so a new domain field cannot
   quietly join outer state and be released on a refused response. `grounded`
   is inert (a bare boolean) but is pinned `False` rather than left `True`.
2. The `formatted_output` replacement is deliberately **non-empty**. A falsy
   stand-in (`""` / `{}`) does not suppress the `or result` fallback — it
   ACTIVATES it, producing exactly the leak it was written to prevent.
3. This template **overrides** `get_output()`, so the un-guarded resolution is
   re-implemented in template code and must be fixed there too: `result` is
   surfaced only when `status == SUCCESS`, and on any non-success outcome
   `output` is re-resolved as `formatted_output or None`. Either half alone
   contains the leak on today's paths; both are required so that no future
   writer of `result` (the inner `OutputFormatNode` already writes one into
   *inner* state) can re-open it.

**Detector parity is part of containment, not a nicety.** The framework scans
every node result with `detect_credentials()` and *raises* on a hit, and a
raise makes `BaseNode.__call__` discard the node's entire return — including
the clearing above. A shape the framework catches and the domain gate misses is
therefore a containment bypass rather than a lenient gate. `_security_gate_output()`
consults the framework detector first and only then its domain-extra patterns,
so the union is a strict superset by construction. `InputValidateNode` imports
the same helper for caller-corpus ingest, so ingest widens with it.

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (`config["configurable"]` — session_id, trust_level)
- [x] `framework.security.credential_detector.detect_credentials()` — the
  framework's own recognizer, consulted by the domain output gate so the two
  cannot drift apart (see "Output boundary contract")
- [x] Module-level `_security_gate_output()` in `post_process_node.py` —
  credential/secret scan on the answer string (also reused at ingest for
  caller passage text/source)
- [x] `emit_trace_event()` — at least one domain-specific event per node `execute()`
- [x] `to_json()` / `from_json()` / `finite_in_range()` helpers in
  `src/schemas/state.py` — one serialisation + numeric-validation contract

### Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Caller-auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.

## Composition Pattern

- **Pattern**: Cat 2 nested two-layer — GraphNode wrapping inner BaseGraph
- **Outer graph**: `PolicyRegulationQAAgent(AgentBaseGraph)` — fixed 5-node backbone
- **Inner graph**: `DomainWorkflowGraph(BaseGraph)` — 5-node linear RAG pipeline
- **input_context hand-off**: ContextVar bridge (`src/graph/context_bridge.py`)
- **Error propagation**: propagate (SubgraphError on inner failure; outer backbone retries pre_process)

## Import Isolation Confirmation
- [x] Template does not import the platform SDK
- [x] Import targets: `framework/` and `shared/` only

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Framework base type | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | Fixed sequential RAG pipeline; no autonomous reasoning loop |
| Composition pattern | Cat 1 (flat) | Cat 2 (nested GraphNode) | Cat 2 nested | Multi-step retrieve→rerank→generate RAG workflow |
| Grounding on low confidence | Best-effort answer | Decline below threshold | Decline | Life/safety-adjacent — never guess a legal/procedural fact |
| Retrieval threshold | 0.73 (generic) | 0.78 (raised) | 0.78 | Raise the grounding bar for regulatory guidance |
| Generation temperature | > 0 | 0.0 | 0.0 | Deterministic, faithful to source policy text |
| State dict fields | bare dict | JSON-serialised str | JSON-serialised str | msgpack serialisation safety |
| LLM integration | Real LLM | Deterministic extractive synthesis | Deterministic (extractive) | The installed framework SDK ships no LLM client; the prompt template + config are the production wiring point |
| Caller corpus vs built-in | Merge both | Replace per request | Replace | Provenance clarity — a request's citations come from one corpus, never a mix |
| Caller threshold override | Free range | Tighten-only (0.78–1.0, floor 0.75) | Tighten-only | A caller must never be able to lower the grounding bar |
