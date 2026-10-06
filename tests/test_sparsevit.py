import torch

from models.sparsevit_backbone import SparseViTBackbone


def test_sparsevit_feature_shapes() -> None:
    model = SparseViTBackbone(
        embed_dims=(32, 64, 160, 256),
        depths=(1, 1, 1, 1),
    ).eval()
    with torch.no_grad():
        features = model(torch.randn(1, 3, 64, 64))
    assert [tuple(value.shape) for value in features] == [
        (1, 32, 16, 16),
        (1, 64, 8, 8),
        (1, 160, 4, 4),
        (1, 256, 2, 2),
    ]
