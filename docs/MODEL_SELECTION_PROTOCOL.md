# ArrestShield Model Selection Protocol

This protocol is fixed before comparative training. It prevents selecting a
model after seeing whichever metric makes it look best.

## Primary rule

Select one decision threshold per seed using validation data only. The threshold
must keep the false-positive rate at or below 5% on the hard-negative validation
set. Among eligible thresholds, maximize scam recall. Compare models using mean
validation recall at that fixed operating rule across seeds `17`, `42`, and
`93`. Evaluate the untouched test views only after selecting the model family.

If two models' mean recall differs by no more than one pooled standard deviation,
treat them as statistically tied. Resolve a tie using, in order:

1. Lower median stable scammer turns-to-detection.
2. Higher macro-F1.
3. Better Roman/Devanagari mixed-script coverage.
4. Smaller deployable artifact.

Accuracy is never a selection metric because the corpus is strongly imbalanced.

## Stable early detection

For a conversation with prefix scores `s1...sn`, let `H` be the selected entry
threshold and `L = 0.8H` the hysteresis exit threshold. Stable detection is the
first scammer turn `t` where `st >= H` and all remaining prefix scores are at
least `L`. Report the median stable scammer turn, its interquartile range, and
the percentage never detected. Also report ordinary non-latched prefix scores
for transparency.

## Required evaluation views

- Conversation-group test: the normal group-safe held-out set.
- Mixed-source test: only sources containing both scam and legitimate examples.
- Hard-negative test: legitimate conversations containing overlapping words or
  contexts such as bank, payment, police, courier, Aadhaar, OTP, fraud, and
  account support.
- Leave-one-mixed-source-out: train without one mixed source and test on that
  source to expose dataset-origin shortcuts.

Source channel and near-duplicate group must never cross a declared holdout
boundary. Audio experiments additionally group by recording channel/uploader so
the detector cannot learn microphone, compression, or creator artifacts.

## Model ladder

1. TF-IDF word/character features with SGD logistic loss.
2. TF-IDF/SVD features with XGBoost.
3. HingRoBERTa-Mixed multi-task transformer candidate.
4. XLM-R, IndicBERT, and MuRIL comparison candidates when GPU compute is available.
5. Transparent weighted risk fusion, followed by an XGBoost fusion comparison
   trained only on out-of-fold detector outputs.

The I4C complaint benchmark is supporting evidence for Hinglish encoders, not
direct evidence for this task: its inputs are retrospective, victim-written,
first-person complaints, while ArrestShield receives live, adversarial scammer
speech. This domain transfer is a hypothesis that must be evaluated.

## ASR coupling

Whisper and IndicWhisper must be evaluated using both transcript WER and final
detection performance. Different transcription scripts and normalization can
change classifier behavior even when WER improves. Audio model selection uses
downstream recall at the same hard-negative FPR, with WER as a supporting metric.
