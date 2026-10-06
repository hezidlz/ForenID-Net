# Pretrained weights

The repository includes the small frozen Noiseprint++ checkpoint used by the
model. Its SHA-256 digest is:

```text
51c153596efecdfc9bdff1b961d63855b943e6a8fd37083cc7499e60b36dd8bf  noiseprint++.th
```

SparseViT does not require a separate external initialization file. The general
forgery pretraining stage creates the checkpoint used for document fine-tuning.

The included pretrained file remains subject to the original TruFor terms in
`third_party/licenses/TruFor-LICENSE.txt`.
