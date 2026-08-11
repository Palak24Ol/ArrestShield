# ArrestShield Annotation Codebook

## Purpose and unit of annotation

Annotate each speaker turn while displaying the preceding conversation. Labels
describe evidence in the current turn, not an annotator's guess about what may
happen later. Each positive label must include an exact supporting span. When
the evidence is ambiguous, use `unknown`; do not silently convert uncertainty
to a negative label.

The codebook uses a hierarchical hybrid. The nine research-level psychological
techniques come from PsyScam so results can be compared with published work.
Six additional operational signals cover digital-arrest behavior that PsyScam
does not model explicitly, especially secrecy and enforced isolation. This is
more defensible than either discarding the published taxonomy or forcing all
digital-arrest behavior into five broad labels.

## Research psychological techniques

| Label | Apply when | Do not apply merely because |
|---|---|---|
| `authority_impersonation` | The speaker claims or invokes police, CBI, RBI, bank, courier, court, government, employer, or another authority/identity to obtain compliance. | An organization is discussed neutrally. |
| `phantom_riches` | The target is promised an implausible, guaranteed, unexpected, or unusually easy gain. | A legitimate refund or salary is discussed without deceptive reward framing. |
| `fear_intimidation` | The speaker threatens arrest, prosecution, blocking, financial loss, reputational harm, violence, or another serious consequence. | A legitimate warning accurately explains a routine policy without coercion. |
| `liking` | Flattery, sympathy, similarity, charm, or relationship-building is used to lower resistance. | The speaker is merely polite. |
| `urgency_scarcity` | A deadline, immediate action, limited opportunity, or artificial shortage pressures the target. | A real appointment time is stated without pressure. |
| `pretext_trust` | A cover story or personal/contextual detail is used to manufacture legitimacy or familiarity. | Context is necessary for an ordinary support interaction. |
| `reciprocity` | A favor, gift, advance, or help is used to create an obligation to comply. | A normal service is provided without a requested return favor. |
| `consistency_commitment` | A small prior action or statement is used to pressure a larger action. | The speaker simply repeats a request. |
| `social_proof` | Claimed approval or behavior of other people is used to normalize the requested action. | Other people are mentioned without persuasive comparison. |

## Digital-arrest operational signals

| Label | Apply when | Example evidence |
|---|---|---|
| `accusation` | The target, Aadhaar, SIM, account, parcel, or relative is alleged to be connected to a crime. | "Aapke Aadhaar se money laundering hui hai." |
| `secrecy_instruction` | The target is told not to disclose the situation to family, bank staff, police, or others. | "Kisi ko mat batana." |
| `isolation_instruction` | The target is told to remain alone, enter a room, avoid other people, or stay continuously connected. | "Room lock karke call par rahiye." |
| `surveillance_control` | The speaker demands video presence, screen sharing, remote access, or continuous environmental control. | "Screen share on rakho." |
| `financial_demand` | The speaker asks for a payment, transfer, deposit, cash withdrawal, gift card, or asset movement. | "Safe account mein paise transfer karo." |
| `credential_request` | The speaker asks for an OTP, PIN, password, CVV, login code, or recovery secret. | "Verification ke liye OTP bataiye." |

## Scam stage

Assign one primary stage to each turn. If two stages occur, choose the most
advanced stage supported by direct evidence and retain all applicable tactic
labels.

1. `contact` - establishes communication or introduces a neutral pretext.
2. `authority_claim` - claims an official identity or institution.
3. `accusation` - links the target or their property/identity to wrongdoing.
4. `threat` - describes punitive consequences or intimidates the target.
5. `isolation_control` - imposes secrecy, isolation, surveillance, or call control.
6. `payment_extraction` - requests money, credentials, or a transaction.
7. `post_payment` - confirms, repeats, launders, or escalates after a transaction.
8. `none_unknown` - no scam stage is supported or the turn is ambiguous.

## Gold-label workflow

Two annotators independently label each turn and attach evidence spans. Exact
agreement becomes provisional gold. Disagreement is reviewed by a third
annotator, who records the final decision and a short rationale. Automated or
LLM-generated suggestions may help annotators find candidate spans, but may
never become gold without this human verification.

Source-gold labels that describe a different task (for example, a customer
service intent label) do not automatically become gold scam labels. Victim-
written retrospective complaints must also be tagged by discourse perspective;
they are not interchangeable with live scammer speech.

## Quality checks

- Report Cohen's kappa for binary/stage labels and per-label agreement for the
  multi-label tactics.
- Audit at least 10% of agreed-negative turns for missed positive evidence.
- Never split turns from one conversation or similarity group across datasets.
- Preserve `speaker_role`, `discourse_perspective`, `source_channel`, script,
  and ASR provenance for slice evaluation.
