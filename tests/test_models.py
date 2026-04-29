import torch

def test_hierdrnet_output_shape():
    import sys; sys.path.insert(0, ".")
    from src.models.train import HierDRNet
    model = HierDRNet(pretrained=False)
    x     = torch.randn(2, 3, 260, 260)
    s1, s2, f = model(x)
    assert s1.shape == (2, 2),  f"Stage1 shape mismatch: {s1.shape}"
    assert s2.shape == (2, 4),  f"Stage2 shape mismatch: {s2.shape}"
    print("✅ HierDRNet output shapes correct")

def test_cascade_output_shape():
    import sys; sys.path.insert(0, ".")
    from src.models.cascade_model import Stage1Net, Stage2Net
    import torch.nn.functional as F
    s1_model = Stage1Net(pretrained=False)
    s2_model = Stage2Net(pretrained=False)
    x        = torch.randn(2, 3, 260, 260)
    s1_logits, _ = s1_model(x)
    s1_prob  = F.softmax(s1_logits, dim=1)[:, 1:2]
    s2_out   = s2_model(x, s1_prob)
    assert s1_logits.shape == (2, 2), f"Stage1 shape: {s1_logits.shape}"
    assert s2_out.shape    == (2, 4), f"Stage2 shape: {s2_out.shape}"
    print("✅ Cascade output shapes correct")
