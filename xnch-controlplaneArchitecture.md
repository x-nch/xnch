xnch — Control Plane System Design
1. System Definition
xnch is a stateful policy enforcement authority — the single chokepoint through which all action requests, memory mutations, and execution commands must pass before they affect the system or the world.
It is not a middleware layer. It is not a wrapper. It is the constitutional layer of the ecosystem — the component whose decisions are final, whose state is authoritative, and whose logs are the ground truth of what the system did and why it was permitted to do so.
Formally: xnch is a request evaluation engine that accepts a structured action proposal, evaluates it against a policy set, current memory state, and system context, and returns a deterministic verdict: ALLOW | BLOCK | MODIFY | DEFER.
It maintains no opinion about what is best. It enforces what is permitted.
Architecture classification: synchronous, stateful, deterministic gateway with async audit emission.

2. Core Modules
2.1 Policy Engine
The hard constraint evaluator. Holds the authoritative ruleset — written as declarative policies (think OPA/Rego-style or a purpose-built DSL), not imperative code. Policies are versioned, immutable once deployed, and auditable. No runtime mutation.
Responsibilities:

Evaluate action proposals against active policy set
Resolve policy conflicts via explicit precedence ordering (no implicit tie-breaking)
Return structured verdict with rule references — never a bare boolean
Support policy dry-run mode for testing without side effects

Policy types it must handle: hard blocks (non-negotiable), conditional allows (requires context), rate/quota enforcement, time-window restrictions, role-capability bindings.
2.2 Memory Store (Control Plane Segment)
xnch does not own all memory — Nexi has its own working context. xnch owns long-term structured memory that is authoritative for governance purposes: decision history, entity state, access records, prior verdicts, memory mutation log.
This is not a cache. It is a write-ahead, append-dominant store with structured schema. No free-form blobs. Every record has: entity ID, timestamp, actor, action type, verdict, policy reference, and a content hash for tamper detection.
Schema must be queryable by the Policy Engine at verdict time — "has this entity triggered this policy class more than N times in window W?" is a policy condition, not a post-hoc query.
2.3 Governance Layer (RBAC)
Role definitions, capability bindings, and actor identity resolution. Every request entering xnch carries an actor identity. The Governance Layer resolves that identity to a role, and that role to a capability set, before the Policy Engine evaluates anything.
Role model: admin | operator | viewer | agent at minimum. Agents (Nexi, execution layer components) are first-class actors with their own capability scopes — they are not elevated humans.
Critical: role assignments are stored in xnch's memory, not passed in the request. A request claiming a role is not trusted. Identity is verified; role is looked up.
2.4 Audit Logger
Async but guaranteed. Every verdict — ALLOW, BLOCK, MODIFY, DEFER — emits a structured audit event before the response is returned to the caller. Not after. Not best-effort. The audit trail must be consistent with the verdict stream.
Audit record schema (minimum):
request_id, timestamp_ns, actor_id, actor_role, action_type,
action_payload_hash, policy_ids_evaluated, verdict, verdict_reason,
memory_snapshot_id, system_state_version
Audit log is append-only, write-once, externally replicated. xnch's own memory and the audit log must be reconcilable — any divergence is a system integrity violation.
2.5 Execution Gate
The final enforcer. Sits between the verdict and the execution layer. Translates ALLOW verdicts into signed execution tokens that the execution layer must present to proceed. MODIFY verdicts rewrite the action proposal before issuing the token. BLOCK verdicts terminate the request with a structured rejection. DEFER verdicts place the action in a held queue pending secondary authorization.
The execution layer never receives a raw action from Nexi. It receives a xnch-signed, time-bounded execution token. This means even if Nexi is compromised, the execution layer has an independent validation path.

3. Interfaces xnch Must Expose to Nexi
These are contracts, not suggestions. Every field is required; partial requests are rejected.
3.1 POST /verdict — Primary Evaluation Interface
Request:
{
  request_id: uuid,
  actor: { id, claimed_role },         // role will be re-resolved, not trusted
  action: {
    type: enum(QUERY | MUTATE | EXECUTE | MEMORY_WRITE | POLICY_CHECK),
    target: string,
    payload_hash: sha256,
    payload: structured_object
  },
  context: {
    session_id: uuid,
    prior_request_ids: [uuid],          // chain for multi-step action tracking
    nexi_reasoning_ref: uuid            // reference to Nexi's decision record
  }
}
Response:
{
  request_id: uuid,
  verdict: ALLOW | BLOCK | MODIFY | DEFER,
  verdict_reason: string,
  policy_refs: [policy_id],
  modified_action: object | null,       // populated only on MODIFY
  execution_token: signed_jwt | null,   // populated only on ALLOW | MODIFY
  token_ttl_ms: int,
  audit_ref: uuid
}
Nexi must not cache verdicts. Every action proposal is a fresh evaluation — system state may have changed between calls.
3.2 POST /memory/read — Governed Memory Query
Nexi cannot query xnch's memory store directly. It requests memory through this interface, and xnch applies read policy before returning anything. This prevents Nexi from pulling sensitive state it is not authorized to reason over.
3.3 POST /memory/write — Structured Memory Mutation
Nexi proposes memory updates. xnch evaluates write policy, applies schema validation, and either commits or rejects. Nexi never writes to xnch's memory directly. All writes go through the verdict path — a memory write is just another action type.
3.4 GET /policy/check — Dry-Run Policy Evaluation
Allows Nexi to test whether a proposed action would pass policy without committing to it. Used for planning — Nexi can evaluate multiple candidate actions before selecting one to formally submit. Returns verdict without issuing execution token and without emitting a full audit event (emits a lighter policy_check trace instead).
3.5 GET /system/state — System State Snapshot
Returns current system state version, active policy set version, and any active holds or overrides. Nexi must call this at session initialization. If system state version has changed mid-session, Nexi must re-evaluate its in-progress reasoning chain.
3.6 POST /audit/query — Audit Access for Nexi Reasoning
Nexi may need prior decision history to reason correctly. It accesses audit history through this governed interface, not through direct memory access. xnch applies actor-scoped filtering before returning audit records.

4. Failure Modes if xnch is Weak or Missing
These are not theoretical. Each one maps to a class of production incident.
4.1 Policy Bypass Through Reasoning
If Nexi can reach the execution layer without going through xnch, sufficiently complex reasoning chains will eventually construct a path that violates policy — not through malice, but through optimization. Probabilistic systems find loopholes. Without xnch as a hard gate, "the model figured out a way to do it" is a production incident waiting to happen. Mitigation requires xnch to be the only path to execution — not the recommended path.
4.2 Audit Inconsistency / Decision Opacity
A weak xnch that logs asynchronously without consistency guarantees will produce an audit trail that diverges from reality under load or failure conditions. When you need to reconstruct why the system took a destructive action, "the audit log might not reflect what actually happened" is an unacceptable answer in any compliance context. Mitigation: audit emission must be synchronous with verdict issuance, transactionally consistent.
4.3 Memory Authority Fragmentation
If memory can be written by multiple components without going through xnch, you lose the ability to enforce memory integrity guarantees. Nexi could write contradictory state. The execution layer could update entity records post-action without policy evaluation. Over time, memory becomes inconsistent and the system's decisions become unrepeatable. xnch must be the sole write authority for governed memory.
4.4 Role Escalation
Without a Governance Layer that resolves roles from internal state, a compromised Nexi instance (or a malformed request from an external agent) can claim elevated permissions. If xnch trusts claimed roles, the entire capability model collapses to the least secure component in the call chain. Mitigation: identity is authenticated, role is always resolved internally, and role resolution is itself audited.
4.5 Execution Without Token Validation
If the execution layer accepts action instructions directly from Nexi — even well-intentioned ones — you have a two-actor system with no enforcement point between reasoning and effect. A signed execution token with a TTL means the execution layer has an independent, xnch-derived proof that the action was evaluated and approved at a specific point in time. Without this, execution is running on trust rather than proof.
4.6 System State Version Skew
Without a versioned system state that Nexi must acknowledge, a Nexi instance that began reasoning under policy version N can complete its reasoning and submit an action after policy version N+1 has been deployed — potentially submitting an action that was legal under the old policy but blocked under the new one, or worse, one that the old policy would have modified but the new policy allows through. xnch must reject execution tokens issued against a superseded system state version.
4.7 Feedback Loop Corruption
If memory updates happen outside xnch's write path, the feedback loop — the mechanism by which the system improves — is writing to unvalidated state. The system then learns from corrupted or unevaluated signals. This is the slowest-moving and hardest-to-detect failure mode: the system gradually drifts toward behavior that reflects its corrupted memory, not its intended policy.



Summary: xnch is not optional infrastructure. It is the component that makes everything else in the ecosystem trustworthy. Every other layer — Nexi, models, execution — operates correctly only because xnch is functioning correctly. Its failure modes are not edge cases; they are the canonical failure modes of ungoverned AI systems that have already caused production incidents across the industry.