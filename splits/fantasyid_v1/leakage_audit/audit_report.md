# FantasyID cross-partition leakage audit

Audit date: 2026-08-29

Group key: filename stem (base identity/card across devices and attacks)
Perceptual-hash candidate threshold: Hamming distance <= 4

## Confirmed checks

- Group overlap across train, development, internal test, and cross-template
  holdout: **0** for every pair of partitions.
- SHA-256 exact duplicates across reporting partitions: **0**.
- Perceptual-hash candidate pairs: **6,641**, representing **232** distinct
  cross-split group pairs; none share a group key.
- Of those candidates, **6,640/6,641** are within the same document template.
  This is expected because large, fixed ID-card backgrounds dominate a 64-bit
  perceptual hash even when portraits and personal fields differ.
- The only cross-template candidate is
  `portugal-NM-1102` versus
  `netherland-flickr_M_31998389253_0a26f25b0d_h` (distance 4). Visual review
  confirms different templates, identities, portraits, text fields, and capture
  chains.
- Visual review of zero-distance same-template candidates likewise confirms that
  the candidates can contain different identities and edited fields. Therefore,
  perceptual-hash proximity is treated as a review trigger rather than proof of
  duplication for this highly templated dataset.

## Interpretation

No exact duplicate or same-base-document leakage was confirmed. The candidate
file is retained in full so that the audit is reproducible. The audit does not
turn the revised partitions into a historically unseen test set: the project had
previously used the official FantasyID train and test CSVs, as documented in the
evaluation protocol.
