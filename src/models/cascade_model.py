import torch
import torch.nn as nn
from torchvision import models


class Stage1Net(nn.Module):
    """Binary screening: DR / No-DR."""
    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(in_feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2),
        )

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return self.classifier(features), features


class Stage2Net(nn.Module):
    """Severity grading (DR-positive only). Receives Stage1 DR probability as extra signal."""
    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(in_feat + 1, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4),
        )

    def forward(self, x, stage1_prob):
        features = self.dropout(self.backbone(x))
        combined = torch.cat([features, stage1_prob], dim=1)
        return self.classifier(combined)
