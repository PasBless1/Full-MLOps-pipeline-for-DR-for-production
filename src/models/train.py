import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from src.entity.config_entity import ModelTrainerConfig
from src.utils.logger import logger
from src.utils.exception import HierDRException


class HierDRNet(nn.Module):
    """
    Two-stage hierarchical EfficientNet-B4.
    Stage 1 head : binary  DR / No-DR        (all images)
    Stage 2 head : severity Mild/Mod/Sev/PDR (DR-positive masked)
    Both heads share a single EfficientNet-B4 backbone.
    """
    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        in_feat = self.backbone.classifier[1].in_features   # 1792
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.stage1_head = nn.Sequential(
            nn.Linear(in_feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2),
        )
        self.stage2_head = nn.Sequential(
            nn.Linear(in_feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4),
        )

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.stage1_head(f), self.stage2_head(f), f


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"HierDRNet — device: {self.device}")

    def _hierarchical_loss(self, s1, s2, labels, criterion):
        loss_s1  = criterion(s1, (labels > 0).long())
        dr_mask  = labels > 0
        loss_s2  = criterion(s2[dr_mask], labels[dr_mask] - 1) if dr_mask.any()                    else torch.tensor(0.0, device=labels.device)
        return loss_s1 + loss_s2

    def train(self, train_loader, val_loader):
        try:
            c         = self.config
            model     = HierDRNet(pretrained=True, dropout=c.dropout).to(self.device)
            scaler    = GradScaler()
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(),    "lr": c.learning_rate * c.backbone_lr_factor},
                {"params": model.stage1_head.parameters(), "lr": c.learning_rate},
                {"params": model.stage2_head.parameters(), "lr": c.learning_rate},
            ], weight_decay=c.weight_decay)
            scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
            criterion  = nn.CrossEntropyLoss(label_smoothing=c.label_smoothing)
            best_val   = float("inf")
            patience   = 0

            logger.info("=" * 55)
            logger.info("HierDR-Net Training (EfficientNet-B4 + Mixed Precision)")
            logger.info("=" * 55)

            for epoch in range(c.epochs):
                model.train(); train_loss = 0.0
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    with autocast():
                        s1, s2, _ = model(imgs)
                        loss = self._hierarchical_loss(s1, s2, labels, criterion)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                    train_loss += loss.item()

                model.eval(); val_loss = 0.0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(self.device), labels.to(self.device)
                        with autocast():
                            s1, s2, _ = model(imgs)
                            val_loss += self._hierarchical_loss(s1, s2, labels, criterion).item()

                scheduler.step()
                avg_t = train_loss / len(train_loader)
                avg_v = val_loss   / len(val_loader)
                logger.info(f"Epoch [{epoch+1:02d}/{c.epochs}] Train: {avg_t:.4f} | Val: {avg_v:.4f}", )

                if avg_v < best_val:
                    best_val = avg_v; patience = 0
                    os.makedirs(os.path.dirname(c.trained_model_path), exist_ok=True)
                    torch.save(model.state_dict(), c.trained_model_path)
                    logger.info(f"  ✓ Best model saved (epoch {epoch+1})")
                else:
                    patience += 1
                    if patience >= c.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}"); break

            model.load_state_dict(torch.load(c.trained_model_path, map_location=self.device))
            logger.info("HierDRNet training complete.")
            return model
        except Exception as e:
            raise HierDRException(e, sys)


if __name__ == "__main__":
    from src.config.config import ConfigurationManager
    from src.data.transformation import DataTransformation
    manager = ConfigurationManager()
    train_loader, val_loader, _ = DataTransformation(
        manager.get_data_transformation_config()
    ).get_data_loaders()
    ModelTrainer(manager.get_model_trainer_config()).train(train_loader, val_loader)
