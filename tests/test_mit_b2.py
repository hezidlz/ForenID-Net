import torch

from models.mit_b2 import MiTB2Backbone


def test_mit_b2_feature_shapes() -> None:
    model = MiTB2Backbone(image_size=64, drop_path_rate=0.0).eval()
    with torch.no_grad():
        features = model(torch.randn(1, 3, 64, 64))
    assert [tuple(x.shape) for x in features] == [
        (1, 64, 16, 16),
        (1, 128, 8, 8),
        (1, 320, 4, 4),
        (1, 512, 2, 2),
    ]
