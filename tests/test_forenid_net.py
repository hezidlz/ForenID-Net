import torch

from models.forenid_net import ForenIDNet


def test_rgb_only_forward_contract() -> None:
    model = ForenIDNet(
        image_size=64,
        noiseprint_weights=None,
        mit_b2_weights=None,
        fusion_mode="rgb_only",
        use_noiseprint=False,
    ).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 64, 64))
    assert output["pred_mask"].shape == (1, 1, 64, 64)
    assert output["pred_score"].shape == (1,)
    assert torch.all((output["pred_mask"] >= 0) & (output["pred_mask"] <= 1))
    assert torch.all((output["pred_score"] >= 0) & (output["pred_score"] <= 1))


def test_learned_gate_with_distributed_noiseprint_weight() -> None:
    model = ForenIDNet(
        image_size=64,
        noiseprint_weights="weights/noiseprint++.th",
        mit_b2_weights=None,
        fusion_mode="learned_gate",
    ).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 64, 64), return_aux=True)
    assert output["pred_mask"].shape == (1, 1, 64, 64)
    assert output["pred_score"].shape == (1,)
    assert len(output["nfa_gates"]) == 4
    assert all(torch.isfinite(value).all() for value in output["nfa_gates"])
    assert torch.equal(output["rgb_features"][0], output["fused_features"][0])
    assert all(
        not torch.equal(output["rgb_features"][index], output["fused_features"][index])
        for index in (1, 2, 3)
    )
