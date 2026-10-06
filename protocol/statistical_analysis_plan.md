# Statistical analysis plan

## Locked endpoints

- Primary image endpoint: ROC AUC.
- Secondary image endpoints: AP, development-calibrated F1, balanced accuracy,
  specificity, MCC, and TPR at FPR = 1%.
- Primary localization interpretation: pointing-game accuracy and enclosing-box
  IoU against box-derived weak labels.
- Secondary localization endpoints: region recall, F1/IoU against box-derived
  masks, and sampled pixel AUC/AP explicitly labelled by annotation provenance.

## Calibration and model selection

- Select one checkpoint per run using development-set image AUC.
- Select image and mask operating thresholds on the development split only.
- If several thresholds achieve the same development F1, choose the largest
  threshold as a conservative false-alarm tie-break.
- Freeze the checkpoint and both thresholds before any revision test inference.
- Do not report test-set `F1-best` or test-selected thresholds.

## Repetition and uncertainty

- Main trainable methods use five seeds: 20260829--20260833.
- Report seed mean and standard deviation.
- Within each seed, compute 2,000-resample 95% percentile intervals using the
  base-document group, not individual captured images, as the resampling unit.
- For model comparisons on the same samples, use the same resampled groups and
  report the paired difference interval. Call an advantage stable only when the
  interval excludes zero and the direction is consistent across seeds.
- Subgroup tables must include samples, base-document groups, real/fake counts,
  and 95% intervals. Attack-only subgroups are localization-only and are not used
  for binary AUC/AP.

## Multiplicity and interpretation

The method has one primary endpoint. Other metrics, subgroup rankings,
degradation curves, and ablations are secondary or exploratory; they are not
used to make multiple independent superiority claims. The manuscript must
distinguish stable advantages from uncertain numerical differences.
