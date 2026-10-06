# ForenID-Net revision evaluation protocol

## Purpose

The original draft used the same FantasyID evaluation set for model selection,
threshold selection, and final reporting. The revision separates these roles and
keeps all samples derived from the same base document in one partition.

## Group definition

The group key is the image filename stem. FantasyID uses the same stem for the
bonafide card, its manipulated variants, and captures from different devices.
Consequently, grouping by stem keeps the following together:

- the same synthetic identity/base card;
- digital_1, digital_2, and digital_3 variants of that card;
- Huawei, iPhone, and scanner captures of that card;
- all masks and metadata attached to those images.

The template stratum is the language/design prefix before the first hyphen.

## Partitions

1. `train`: 70% of the 211 base-document groups in the official training CSV.
2. `dev`: 15% of those groups, used for checkpoint and threshold selection only.
3. `internal_test`: 15% of those groups, not used for model or threshold selection
   in the revision runs.
4. `external_clean_test`: official-test groups that do not occur in the official
   training CSV and that contain both bonafide and attacked samples.

The development and internal-test lists also include digital_3 samples belonging
to their own held-out groups. Digital_3 samples belonging to training groups are
excluded from evaluation because they share the same base documents as training.
Attack-only official-test groups without matched bonafide samples are excluded
from complete binary comparisons and may be used only for conditional localization
analysis, if clearly labelled.

### Retrospective-exposure limitation

This split is group-disjoint for the revision runs, but it is not a genuinely
project-blind holdout. Earlier exploratory models were trained on the full official
FantasyID training CSV, and the official test CSV was previously used for reporting.
Accordingly, the revised FantasyID results must be labelled a *retrospective
group-disjoint re-evaluation*. A never-seen confirmatory claim requires a new
external dataset (for example, an authorised DocTamper subset or a held-out MIDV-DM
subset) that has not been used in any model or design decision.

## Selection and reporting

- Select checkpoints on `dev` using image AUC as the predeclared primary ranking
  metric; use region AP against box-derived masks only as a reported secondary
  endpoint, not as an additional selection opportunity.
- Select image and region operating thresholds on `dev` only.
- Apply the frozen thresholds once to `internal_test` and `external_clean_test`.
- Report AUC and AP as threshold-independent metrics.
- Report F1, accuracy, precision, recall, specificity, balanced accuracy, MCC,
  and TPR at 1% FPR using development-set calibration.
- Run five independent seeds (`20260829` through `20260833`) and report mean,
  standard deviation, base-document-group bootstrap 95%
  confidence intervals, and paired bootstrap differences for main comparisons.
- Treat box-derived masks as region-level weak supervision. Pixel metrics must be
  labelled `against box-derived masks`; region recall, pointing-game accuracy,
  and box IoU are the primary localization interpretations.
- The final ForenID-Net variant is predeclared as the heavy-degradation version.
  This choice was made from the earlier exploratory run and must not be revisited
  using either revision test partition.

## Generalization analyses

- Leave-one-template-out evaluation across the ten training template strata.
- Leave-one-device-out and leave-one-attack-family-out evaluation where both
  classes remain defined.
- Zero-shot and equal-budget FantasyID fine-tuning results for trainable baselines.
- A legally usable external document-manipulation dataset when available.
- Multi-level compression, blur, downsampling, moire/scan, and noise-matching tests.

## Integrity checks

- Zero group-key overlap across train/dev/internal-test.
- SHA-256 exact-duplicate audit across all reporting partitions.
- Perceptual-hash near-duplicate audit with all candidate pairs reviewed.
- Public split manifest containing paths, group keys, template strata, devices,
  attack families, labels, and mask provenance.
