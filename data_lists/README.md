# Generated data lists

This directory is intentionally distributed without machine-specific dataset
paths. Generate the list files after placing each dataset on the local machine:

```bash
python scripts/prepare_imd2020.py --help
python scripts/prepare_casia_v2.py --help
python scripts/prepare_fantasticreality.py --help
python scripts/prepare_general_pretrain.py --help
```

The final command produces `general_pretrain_train.txt` and
`general_pretrain_valid.txt`, which are referenced by
`configs/pretrain_general.yaml`. FantasyID uses the versioned, group-disjoint
lists under `splits/fantasyid_v1/lists/`.
