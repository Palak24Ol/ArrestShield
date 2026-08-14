# Training and results

## Data used

The packaged model uses every cleaned, locally available source relevant to
transcript content:

1. Public real scam-call transcripts (source-URL grouped adaptation split).
2. Indian Hinglish scam-call conversations.
3. Indian multilingual scam messages.
4. Synthetic multi-agent scam conversations.
5. Synthetic scam dialogues.
6. BANKING77 legitimate banking hard negatives.
7. DailyDialog legitimate conversations.
8. Schema-Guided Dialogue legitimate service/support conversations.
9. A small mixed-label project-authored hard-example set teaching scam demands
   versus legitimate warnings.

Exact duplicates in the canonical pipeline are removed, and conversation groups
do not cross its train/validation/test partitions. Sources with unclear licences
or unrelated labels are not silently included.

## Model

- Input: one transcript string.
- Features: word 1-2 grams and character 3-5 grams using TF-IDF.
- Classifier: linear SGD with logistic loss.
- Seeds evaluated: 17, 42, and 93.
- Packaged seed: 42.
- Output: scam score plus fixed threshold, producing `SCAM` or `NOT_SCAM`.
- Model size: approximately 3 MB.

## Main demo result

The fixed project-authored behavioral test contains ten scam and ten legitimate
English/Hinglish cases. The packaged model scores **18/20 (90%)**:

- scam recall: 10/10;
- legitimate specificity: 8/10;
- errors: two legitimate safety/awareness statements were conservatively
  flagged as scams.

This 90% result is the clearest number for the professor demo. It is a small
behavioral test, not a production benchmark. The public-corpus split produces
higher scores but shares corpus characteristics with training data, so it must
not be described as real-world accuracy.

## Known limits

- Short or vague scam openings may lack enough evidence.
- Fraud-awareness messages can contain the same vocabulary as scammers and
  cause false alarms.
- Real noisy Hindi/Hinglish audio needs a larger consented validation set.
- The system is a research demonstration, not permission to intercept or place
  real telephone calls.
