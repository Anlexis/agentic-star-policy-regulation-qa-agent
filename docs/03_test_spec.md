# Test Specification — GOV-C2-001 Policy & Regulation Q&A Agent

## 1. Test Strategy

- **Agent:** GOV-C2-001 — Policy & Regulation Q&A Agent (Cat 2, RAG pattern,
  two-layer nested graph: outer `AgentBaseGraph` backbone + inner
  `DomainWorkflowGraph` `BaseGraph`).
- **Coverage target:** ≥ 90% of `src/nodes/` + `src/graph/` branches.
- **Test types:** Unit (per node + graph wiring + caller-data contract +
  runtime-config plumbing) · Proof-of-Boundary (framework security /
  serialization contracts) · End-to-end (`POST /invoke` through the real ASGI
  app, and full `Graph().invoke()`).
- **Framework provisioning:** `framework` (agenticstar-agentcore) is installed
  from the package registry (`agenticstar-agentcore==1.0.1`). Tests import the
  REAL production modules; there are no stub nodes.
- **Audit events:** `emit_trace_event` is patched at the node module level in
  unit tests to avoid audit-backend calls, never via a `sys.modules` stub
  (which would break the real `shared` package the framework loads at import
  time).

### RAG grounding discipline (GOV — life/safety-adjacent)

Government policy guidance is life/safety-adjacent, so the grounding bar is
raised: `config/config.yaml` pins `retrieval.score_threshold: 0.78`, and
`RerankFilterNode` additionally floors any configured threshold at **0.75**
(`_MIN_ALLOWED_THRESHOLD`). A passage may ground an answer only when it clears
that bar; when nothing clears it, `GenerateAnswerNode` **declines** (abstains)
rather than fabricating regulatory guidance. Both the grounded path and the
abstention path are exercised end-to-end. A caller `score_threshold` override
may only TIGHTEN the bar (0.78–1.0, finite) — loosening is structurally
impossible (`max()` + floor).

### Test file map

| File | Scope |
|------|-------|
| `tests/unit/test_nodes.py` | All backbone/domain nodes + outer & inner graph wiring + trust gate + both output-gate layers; output-gate DETECTOR PARITY with the framework recognizer (parametrized over every shape it refuses, with statutory prose as the opposite-direction control) and BLOCK CONTAINMENT (every output-bearing field cleared, inventory pinned against `merge_output()`, replacement non-empty so the `formatted_output or result` fallback cannot resolve to the pre-gate value, clearing survives the framework output scan) |
| `tests/unit/test_output_envelope.py` | The outer `invoke()` envelope (`PolicyRegulationQAAgent.get_output()`): success control (gated answer + structured keys + `result`) vs. every non-success status (ERROR / TIMEOUT / CANCELLED / RETRY / PENDING) — `result` withheld, the base `formatted_output or result` fallback re-resolved so an absent gate output stays absent, `answer_document` never a surfaced key |
| `tests/unit/test_main_node.py` | Deprecated `MainNode` stub — `execute()` contract kept green |
| `tests/unit/test_caller_data_contract.py` | Full caller-data validation matrix: passage bounds/inert ids/ingest credential scan, retrieval overrides incl. the per-field non-finite matrix, channel lockdown, corpus replacement, tighten-only threshold |
| `tests/unit/test_runtime_config.py` | `config/config.yaml` plumbing: values reach the inner nodes end-to-end (config change flips the grounding outcome) |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | TC-06/07 — framework security gates non-bypassable |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | PB-6 per-node + backbone invoke order (VERIFIED_EXTERNAL) + trust-gate denial + payload alignment |
| `tests/proof_of_boundary/test_invoke_e2e.py` | End-to-end `POST /invoke` (real ASGI, Bearer auth): caller-corpus grounding, baseline, abstention, validation rejections (incl. raw JSON NaN/Infinity), size cap, auth, output-schema scan; OUTPUT-GATE CONTAINMENT on the real surface — a clean-path control proves the same request does produce a grounded cited answer, then a gate block (fault injected on the DATA path: a caller question carrying a shape the domain gate refuses but the framework does not, so the domain gate really is the component under test) returns an error envelope carrying no document, no corpus content, no citation label, no `result`, no refused caller string, no traceback and no source path — with `PostProcessNode` in `node_history`, proving the block happened at the gate and not upstream — plus an unpatched companion in which framework-shaped caller data is refused and the envelope carries nothing |
| `tests/proof_of_boundary/test_import_isolation.py` | PB-4 platform-SDK import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | PB-2/PB-5 State msgpack/credential safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | PB-7 HITL interrupt-propagation (skip stub — no cross-boundary HITL) |

### Canonical valid payload (PB-6 `_VALID_PAYLOAD`)

GOV-C2-001 is a RAG agent — the input is a natural-language policy question (a
plain string), not a JSON object. The backbone invoke test and
`deploy/invoke_payload.json` share the same question (the two MUST stay
identical — asserted by `test_invoke_payload_matches_pb6`):

```
What is the filing deadline period for an administrative appeal?
```

Grounding: the question's content terms (administrative / appeal / filing /
deadline / period) fully cover `GOV-KB-002` (行政不服審査法 / Administrative
Appeal Act), giving a retrieval score of 1.0 ≥ 0.78 ⇒ `grounded = True`, so a
grounded, cited answer is produced (never a decline).

## 2. Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected Result | Where |
|-------|------|----------------|-------|
| TC-01 | State contract: flat `TypedDict`, dict/list fields JSON-serialized `Optional[str]`, no Pydantic/dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Invalid/empty/over-length question rejected at PreProcessNode | `status=error`, error_log populated | `TestPreProcessNode` |
| TC-03 | No JWT/credential in State | credential scan: 0 violations | `test_state_safety.py` + repository gate scripts |
| TC-04 | `execute(self, state)` contract — no `_invoke_impl` | Signature `(self, state)`, `_invoke_impl` absent | `test_execute_signature_is_state_first`, `test_main_node.py` |
| TC-05 | Audit: `emit_trace_event()` called inside each node `execute()` | ≥1 domain event per node (positional form) | node implementations (asserted indirectly by patched-emit fixtures) |
| TC-06/07 | Framework input/output security gates cannot be overridden | TypeError at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | `required_trust_level` enforced in `__call__` before `execute()` | ANONYMOUS caller → refused; VERIFIED_EXTERNAL → admitted | `TestTrustGate` |
| TC-08a | Outer `PreProcessNode` = VERIFIED_EXTERNAL; inner nodes + post_process = ANONYMOUS | trust levels asserted per node | `test_trust_level_*` |
| TC-11 | Output gate on post_process | credential pattern → redacted + `status=error`; internal-state blob → `[REDACTED]`; clean/statutory content → byte-identical pass | `TestPostProcessNode` |

## Marketplace Entry Point — `tests/unit/test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `PolicyRegulationQAAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.

## 3. Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Test | Expected Result | Where |
|-------|----------|------|----------------|-------|
| PB-2 | State serialization | AST scan of `src/schemas/state.py` | primitives only; no Pydantic/dataclass | `test_state_safety.py` |
| PB-4 | Import isolation | AST scan of `src/` | 0 platform-SDK imports | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named fields / prohibited types in State | inspection pass | `test_state_safety.py` **Auto-waived — checkpointing disabled**: `config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`, so no checkpoint surface exists; the conditional gate and the non-lossy traversal helper ship with the stub. |
| PB-6 | Invoke execution order (per node) | `__call__`: node_start → trust gate → input gate → `execute()` → output gate → node_complete | order verified for every `src/nodes/` class | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | full `Graph().invoke(_VALID_PAYLOAD, ctx=VERIFIED_EXTERNAL)` | `status=success`; node_history = `[Initialize, PreProcess, PolicyQAGraphNode, PostProcess, Finalize]` | `TestBackboneInvokeOrder` |
| PB-6c | Real external caller | `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` — **never** `for_internal()` | inner ANONYMOUS nodes accept the passthrough trust; SUCCESS end-to-end | `TestBackboneInvokeOrder` |
| PB-6d | Payload alignment | `deploy/invoke_payload.json["input"] == _VALID_PAYLOAD` | deploy verification exercises the PB-6 payload | `test_invoke_payload_matches_pb6` |
| PB-7 | HITL interrupt propagation | skip stub — `propagate_hitl=False`, no cross-boundary interrupt() checkpoint | skipped with reason (real assertion when HITL wired) | `test_pb7_hitl_interrupt_propagation.py` |
| PB-E2E | `/invoke` boundary | real ASGI app, Bearer auth | caller data → real grounded output; malformed data → rejection without echo; oversized context → 413; missing token → 401; output-schema scan clean | `test_invoke_e2e.py` |
| PB-E2E-C | Output-gate containment at `/invoke` | real ASGI app; fault injected on the DATA path (caller question trips the domain output ceiling) | clean-path control releases the grounded answer; the blocked run returns `status=error` with `result=None`, `output` = the gate's own non-empty withholding notice, every structured key `None`, `PostProcessNode` in `node_history`, and no document / corpus text / citation id / refused string / traceback anywhere in the body | `TestBlockedOutputIsContained` |

## 4. Business Logic Tests

| BL-ID | Test | Input | Expected Result | Where |
|-------|------|-------|----------------|-------|
| BL-01 | Happy-path grounded Q&A | `_VALID_PAYLOAD` | grounded, cited answer; `POLICY & REGULATION Q&A` + `Grounded: YES` present | `test_backbone_invoke_succeeds_and_returns_output`, `test_inner_graph_invoke_grounds_and_cites` |
| BL-02 | Retrieval ranking | appeal-deadline question | top candidate `GOV-KB-002`, score ≥ 0.78; top_k respected | `TestRetrieveNode` |
| BL-03 | Grounding bar filters low-confidence passages | scored candidate list | only passages ≥ threshold survive; `grounded` reflects survivor count | `TestRerankFilterNode` |
| BL-04 | Life/safety threshold floor | config `score_threshold=0.5` | floored to 0.75; a 0.60 passage stays excluded | `test_life_safety_floor_overrides_low_config_threshold` |
| BL-05 | Grounded synthesis + citations | filtered passage(s) | answer cites `GOV-KB-002`; `citations` carries `{id, source}` | `test_grounded_answer_cites_passages` |
| BL-06 | Abstention discipline (node) | empty filtered set | `GenerateAnswerNode` returns the decline message; empty citations | `test_declines_when_no_grounded_passages` |
| BL-07 | Abstention discipline (end-to-end) | off-domain question | `grounded=False`, `Grounded: NO` in the document | `test_inner_graph_invoke_declines_when_ungrounded`, `test_off_domain_question_declines` |
| BL-08 | Cited-document assembly | answer + citations + grounded | `POLICY & REGULATION Q&A` header, cited sources, `Grounded: YES/NO` | `TestOutputFormatNode` |
| BL-09 | Graph key coupling | inner `get_output` ↔ outer `merge_output` | 6 coupled keys mapped; `merge_output` returns changed keys only; `grounded` defaults `False` | `TestOuterGraphComposition`, `TestInnerDomainGraph` |
| BL-10 | Caller corpus grounds a real answer | caller `policy_passages` + matching question | grounded cited answer naming the caller passage id — through `/invoke` | `test_caller_corpus_produces_grounded_cited_answer` |
| BL-11 | Corpus replacement (no mixing) | caller corpus present | citations name only caller ids; built-in ids absent | `test_caller_corpus_replaces_builtin`, `test_output_schema_scan_on_grounded_response` |
| BL-12 | Config values live end-to-end | tightened `score_threshold` in `config/config.yaml` | same question flips grounded → decline | `TestConfigReachesInnerGraphEndToEnd` |

### Negative / boundary cases

| Case | Node | Expected |
|------|------|----------|
| empty `user_input` | PreProcessNode | `status=error`, "empty" |
| non-string `user_input` | PreProcessNode | `status=error` |
| question > 2000 chars | PreProcessNode | `status=error`, "exceeds" |
| non-inert `channel` label | PreProcessNode | `status=error`, field named, value not echoed |
| non-dict `input_context` | PreProcessNode / InputValidateNode | `status=error` |
| empty / whitespace `validated_input` | InputValidateNode | `status=error`, "empty" |
| query < 3 chars | InputValidateNode | `status=error`, "shorter" |
| control chars in query | InputValidateNode | stripped; clean `query` |
| passage list empty / >40 / non-list | InputValidateNode | `status=error`, field named |
| passage id non-inert / duplicate | InputValidateNode | `status=error`, field named, value not echoed |
| passage text <20 / >4000 / non-string | InputValidateNode | `status=error`, field named |
| credential pattern in passage text/source | InputValidateNode | `status=error` at ingest (fail closed) |
| keywords non-inert / >15 | InputValidateNode | `status=error`, field named |
| `top_k` bool / float / 0 / 11 / NaN | InputValidateNode | `status=error` |
| `score_threshold` NaN / ±Infinity / string / out-of-range (per-field matrix) | InputValidateNode | `status=error` — fail CLOSED, never a silently-disabled bar |
| missing `query` | RetrieveNode | `status=error`, "missing" |
| off-domain query | RetrieveNode | all candidate scores < 0.78 |
| malformed stored override reaching RetrieveNode / RerankFilterNode | both | ignored — configured default stays in force (defense in depth) |
| empty candidate set | RerankFilterNode | `grounded=False`, empty `filtered_docs` |
| no grounded passages | GenerateAnswerNode | decline message, empty citations, `status=success` |
| empty `answer` | OutputFormatNode | fallback text, `status=success` |
| empty `answer_document` | PostProcessNode | fallback message, `status=success` |
| credential leak in output | PostProcessNode | redacted stub, `status=error` |
| internal-state blob embedded verbatim | PostProcessNode | `[REDACTED]` substitution + audit event |
| statutory content (deadlines, article numbers, years) | PostProcessNode | byte-identical pass-through |
| credential shape the FRAMEWORK refuses (`sk_live_`, `AKIA…`, dot-less `eyJ…`, conn string) | PostProcessNode / InputValidateNode ingest | blocked by the same helper — the gate delegates to `detect_credentials()`, so it can never be narrower than the scan the framework enforces one layer up |
| a blocked response | PostProcessNode + `get_output()` | `formatted_output` = non-empty withholding notice; `result` / `answer` / `answer_document` / `citations` / `filtered_docs` cleared, `grounded` pinned False; envelope surfaces `result` only on the gated success path and re-resolves `output` as `formatted_output or None` on any non-success |

## 5. Test Execution Summary

> **Pending re-run.** The figures below predate the Marketplace entry point work.
> The entry-point test and the PB-5 / PB-6 additions were added after this run and
> have not been executed locally — the framework wheel is not installed in the
> authoring environment. **They are not yet verified anywhere**; this summary is
> updated once a pipeline run has executed them.

- Execution: `python -m pytest tests/ -v` under the real framework wheel
  (the version central CI installs — see the pipeline log, not this document).
- Total: 219 tests — **218 passed, 1 skipped** (PB-7 skip stub, by design).
- Repository gate scripts (manifest schema, dependency pinning, stub check,
  category consistency, credential scan, trust level, forbidden strings) — all
  PASS.
- Coverage: node + graph modules exercised on grounded, abstention, caller-data
  and rejection paths, plus every negative branch above.
