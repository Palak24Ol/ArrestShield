# ArrestShield ML architecture

## Runtime flow

```mermaid
flowchart LR
    A["Live call audio or text turns"] --> B{"Input type"}
    B -->|Audio| C["Local Whisper-family ASR"]
    B -->|Text| D["Conversation formatter"]
    C --> D
    D --> E["Trained multilingual base detector"]
    D --> F["Deterministic privacy-aware entity extraction"]
    D --> G["Trained XGBoost auxiliary heads: scam type, tactics, stage"]
    E --> H["XGBoost risk fusion"]
    F --> H
    D --> H
    H --> I["Frozen threshold plus hysteresis policy"]
    I --> J{"Production-eligible and threshold crossed?"}
    J -->|No| K["Warn, continue observation, or human review"]
    J -->|Yes| L["Policy-approved handoff event"]
    L --> M["Separate LLM/RAG honeypot engagement service"]
    M --> N["Threat-intelligence collection"]

    O["LLM is prohibited from scam scoring"] -.-> E
    O -.-> H
    O -.-> I
```

The ML subsystem owns transcription, feature extraction, trained-model scores, thresholding, and the structured handoff event. It never calls the LLM. The honeypot is a separate downstream service and may receive a handoff only after an eligible trained detector crosses its frozen operating threshold and deployment policy permits engagement.

## Training and evaluation flow

```mermaid
flowchart TD
    A["Licensed public datasets plus future consented project corpus"] --> B["Canonical conversation schema"]
    B --> C["Exact/template deduplication and conversation-group split"]
    C --> D["Train split"]
    C --> E["Validation split"]
    C --> F["Test split: supporting only after selection"]
    D --> G["TF-IDF plus SGD and SVD-XGBoost baselines"]
    D --> H["Causal-prefix XGBoost multi-task heads"]
    D --> P["Optional multilingual transformer comparison"]
    D --> I["OOF base scores plus XGBoost risk fusion"]
    E --> J["Select threshold at hard-negative FPR at most 5%"]
    G --> J
    H --> J
    P --> J
    I --> J
    J --> K["Strict leave-one-source-out shortcut audit"]
    K --> F
    F --> L["Frozen report and artifact hashes"]
    M["Future independently annotated human-gold set"] --> N["Final promotion gate"]
    L --> N
```

## Trust boundaries

- Raw audio is size/type/duration checked, written only to a temporary directory, and deleted after the request.
- Sensitive entities are redacted by default. Raw values require explicit trusted-caller opt-in.
- Model checkpoints and user data are not committed to Git. Compact hashes, configuration, and metrics are committed.
- Current positive labels are entirely silver and mostly synthetic; all checked-in deployment policy therefore remains `research_only_not_promoted`.
- The frozen human-gold collection gate cannot be bypassed by synthetic data, an LLM annotation, or a high random-split score.

The laptop deployment uses the compact classical path: class-weighted SGD for the base probability, XGBoost risk fusion for the research decision score, and separate XGBoost heads for scam type, stage, and supported tactics. The auxiliary heads cannot modify `is_scam`. A transformer implementation remains available as a future controlled comparison, not as a hidden runtime dependency.
