# Pretrained weights

The repository includes the small frozen Noiseprint++ checkpoint used by the
model. Its SHA-256 digest is:

```text
51c153596efecdfc9bdff1b961d63855b943e6a8fd37083cc7499e60b36dd8bf  noiseprint++.th
```

Download the MiT-B2 initialization from the official TruFor repository and
verify its digest with:

```bash
python scripts/download_pretrained_weights.py mit_b2
```

Expected MiT-B2 SHA-256:

```text
ced22617efb7bae3c34ad0a80f20a9b8afb4d27368cb0835a23456baa9d0e092  segformers/mit_b2.pth
```

The pretrained files remain subject to the original TruFor terms in
`third_party/licenses/TruFor-LICENSE.txt`.
