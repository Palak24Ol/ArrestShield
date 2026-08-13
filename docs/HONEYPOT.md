# ArrestShield Honeypot

A post-detection fake-victim service. It engages a caller only after a trained
detector has already decided, and it can never change that decision.

## Boundary

The honeypot is downstream of detection and structurally cannot re-enter it:

- `src/arrestshield/honeypot.py` imports no detector module. A test asserts this.
- The only way in is a signed handoff event produced from a detector response.
- Every transcript is stamped `excluded_from_detector_training: true` and
  `usable_as_detector_evidence: false`.
- No honeypot output is a feature, label, threshold, or score anywhere.

Orchestration happens at the edge, in `scripts/run_honeypot.py`, which imports
both sides. The honeypot library itself stays detector-blind.

## The gate

Default posture is refusal. `configs/deployment/honeypot.json` ships with
`enabled: false`, so a fresh checkout will not engage anyone.

| Condition | Result |
|---|---|
| Detector did not flag scam | blocked — `detector_did_not_flag_scam` |
| Score below `minimum_scam_score` | blocked — `scam_score_below_minimum` |
| `enabled: false`, no research mode | blocked — `honeypot_disabled_by_policy` |
| `enabled: false`, research mode on | allowed, mode `research_only` |
| `enabled: true`, detector not promoted | blocked — `detector_is_research_only` |
| `enabled: true`, detector not promoted, research mode | allowed, mode `research_only` |
| `enabled: true`, detector promoted | allowed, mode `live` |

A non-scam decision blocks engagement even with every switch on. That is
deliberate: the gate's job is to make "engage a real person by mistake" require
more than one wrong flag.

Because the current detector is `research_only_not_promoted`, `live` mode is
unreachable today. `--research-mode` is the explicit operator override for demos.

## Signed handoff

Events are signed HMAC-SHA256 over a fixed field set (`event_id`,
`conversation_id`, `issued_at_utc`, `scam_score`, `threshold`, `detector_status`,
`production_eligible`, `decision_source`), so nothing can forge a "this is a
scammer" event or tamper with a score to clear the gate. Events older than
`maximum_event_age_seconds` (default 300) are rejected, which stops replay of a
stale approval.

Set the secret once:

```bash
setx ARRESTSHIELD_HANDOFF_SECRET "your-long-random-string"
```

Without it, `run_honeypot.py` generates an ephemeral per-run secret. That is fine
for a single-process demo and useless across services — set a real one before
splitting the honeypot onto its own port.

## Synthetic identity

The persona is given exactly four identifiers, and each is built to fail its real
validation rule so a transcript can never contain a live person's details:

| Field | Construction | Why it cannot be real |
|---|---|---|
| Phone | `+91 0XXXX XXXXX` | Indian mobile numbers never start with 0 after the country code |
| Aadhaar-style | 12 digits, checksum forced wrong | Real Aadhaar satisfies a Verhoeff checksum; this deliberately does not |
| Account | `0000` + 6 digits | Reserved documentation-style prefix |
| UPI | `name.NNNN@invalid` | `.invalid` is a reserved non-resolvable TLD (RFC 2606) |

Identifiers are deterministic per conversation, so a caller who asks twice gets
the same answer. `SyntheticIdentity.is_synthetic()` re-checks all four at session
start and refuses to open a session if any check fails.

## Output safety

Model output passes through a redactor before it reaches the caller. Anything
resembling an identifier the persona was *not* given — a real-format Indian
mobile, a 12-digit block, a resolvable email, a 9–18 digit account — is replaced
with a placeholder and logged in the audit trail. The four synthetic values are
allow-listed so they survive.

The system prompt forbids inventing identifiers, completing or promising real
payments, and producing content harmful to anyone. If the caller stops behaving
like a scammer or asks the persona to help defraud a third party, the model
returns `HONEYPOT_DISENGAGE` and the session closes.

Replies are capped at `maximum_reply_characters`, sessions at `maximum_turns` and
`maximum_session_seconds`.

## Setup

Put your Groq key in a git-ignored `.env` at the repository root:

```
GROQ_API_KEY=gsk_your_key_here
ARRESTSHIELD_HANDOFF_SECRET=some-long-random-string
```

`.env` is already in `.gitignore`. The key is read from the environment at call
time, never stored on an object, logged, or written into a transcript.

Model is configurable in `configs/deployment/honeypot.json` under `llm.model`
(default `llama-3.3-70b-versatile`). If Groq retires that model id, change it
there — nothing else needs to move.

## Running it

Show the gate decision without calling the LLM:

```bash
python scripts/run_honeypot.py --demo --dry-run
```

Full engagement on the built-in digital-arrest sample:

```bash
python scripts/run_honeypot.py --demo --research-mode --save reports/honeypot_demo.json
```

Against your own transcript (`[{"speaker_role": "caller", "text": "..."}, ...]`):

```bash
python scripts/run_honeypot.py --transcript call.json --research-mode
```

## Limitations

- Engagement quality is unmeasured. Published honeypot work reports Information
  Disclosure Rate and Human Acceptance Rate; neither is measured here yet.
- The persona is one fixed elderly-victim profile. Real deployments vary persona
  by scam type.
- `.invalid` UPI handles and impossible phone numbers are safe but detectable. A
  sophisticated scammer may notice and disengage. Safety was chosen over realism.
- No rate limiting or abuse controls on the Groq call beyond retry caps.
