import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from src.models.cascade_model import Stage1Net, Stage2Net
from src.entity.config_entity import CascadeModelConfig
from src.utils.logger import logger
from src.utils.exception import HierDRException


class CascadeTrainer:
    def __init__(self, config: CascadeModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Cascade — device: {self.device}")

    def _make_optimizer(self, model, lr, backbone_lr_factor, weight_decay):
        return optim.AdamW([
            {"params": model.backbone.parameters(),    "lr": lr * backbone_lr_factor},
            {"params": model.classifier.parameters(),  "lr": lr},
        ], weight_decay=weight_decay)

    def _fit_stage1(self, model, train_loader, val_loader):
        c         = self.config
        scaler    = GradScaler()
        optimizer = self._make_optimizer(model, c.learning_rate, c.backbone_lr_factor, c.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=c.label_smoothing)
        best_val  = float("inf"); patience = 0
        os.makedirs(c.stage1_model_dir, exist_ok=True)

        logger.info("=" * 55)
        logger.info("STAGE 1 — Binary Screening (DR / No-DR)")
        logger.info("=" * 55)

        for epoch in range(c.epochs):
            model.train(); train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with autocast():
                    logits, _ = model(imgs)
                    loss = criterion(logits, (labels > 0).long())
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
                        logits, _ = model(imgs)
                        val_loss += criterion(logits, (labels > 0).long()).item()

            scheduler.step()
            avg_t = train_loss / len(train_loader)
            avg_v = val_loss   / len(val_loader)
            logger.info(f"[S1] Epoch [{epoch+1:02d}/{c.epochs}] Train: {avg_t:.4f} | Val: {avg_v:.4f}")

            if avg_v < best_val:
                best_val = avg_v; patience = 0
                torch.save(model.state_dict(), c.stage1_model_path)
                logger.info(f"  ✓ Stage 1 saved (epoch {epoch+1})")
            else:
                patience += 1
                if patience >= c.early_stopping_patience:
                    logger.info(f"  Stage 1 early stop at epoch {epoch+1}"); break

        model.load_state_dict(torch.load(c.stage1_model_path, map_location=self.device))
        return model

    def _fit_stage2(self, stage1_model, model, train_loader, val_loader):
        c         = self.config
        stage1_model.eval()
        for p in stage1_model.parameters():
            p.requires_grad = False

        scaler    = GradScaler()
        optimizer = self._make_optimizer(model, c.learning_rate, c.backbone_lr_factor, c.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=c.label_smoothing)
        best_val  = float("inf"); patience = 0
        os.makedirs(c.stage2_model_dir, exist_ok=True)

        logger.info("=" * 55)
        logger.info("STAGE 2 — Severity Grading (DR-Positive Only)")
        logger.info("Stage 1 frozen — DR probability fed into Stage 2")
        logger.info("=" * 55)

        for epoch in range(c.epochs):
            model.train(); train_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                dr_mask = labels > 0
                if not dr_mask.any():
                    continue
                dr_imgs   = imgs[dr_mask]
                dr_labels = labels[dr_mask] - 1
                with torch.no_grad():
                    s1_logits, _ = stage1_model(dr_imgs)
                    s1_prob = F.softmax(s1_logits, dim=1)[:, 1:2]
                optimizer.zero_grad()
                with autocast():
                    loss = criterion(model(dr_imgs, s1_prob), dr_labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                train_loss += loss.item()

            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    dr_mask = labels > 0
                    if not dr_mask.any():
                        continue
                    dr_imgs = imgs[dr_mask]; dr_labels = labels[dr_mask] - 1
                    s1_logits, _ = stage1_model(dr_imgs)
                    s1_prob = F.softmax(s1_logits, dim=1)[:, 1:2]
                    with autocast():
                        val_loss += criterion(model(dr_imgs, s1_prob), dr_labels).item()

            scheduler.step()
            avg_t = train_loss / len(train_loader)
            avg_v = val_loss   / len(val_loader)
            logger.info(f"[S2] Epoch [{epoch+1:02d}/{c.epochs}] Train: {avg_t:.4f} | Val: {avg_v:.4f}")

            if avg_v < best_val:
                best_val = avg_v; patience = 0
                torch.save(model.state_dict(), c.stage2_model_path)
                logger.info(f"  ✓ Stage 2 saved (epoch {epoch+1})")
            else:
                patience += 1
                if patience >= c.early_stopping_patience:
                    logger.info(f"  Stage 2 early stop at epoch {epoch+1}"); break

        model.load_state_dict(torch.load(c.stage2_model_path, map_location=self.device))
        return model

    def train(self, train_loader, val_loader):
        try:
            s1 = Stage1Net(pretrained=True, dropout=self.config.dropout).to(self.device)
            s1 = self._fit_stage1(s1, train_loader, val_loader)
            s2 = Stage2Net(pretrained=True, dropout=self.config.dropout).to(self.device)
            s2 = self._fit_stage2(s1, s2, train_loader, val_loader)
            logger.info("Full cascade training complete.")
            return s1, s2
        except Exception as e:
            raise HierDRException(e, sys)


if __name__ == "__main__":
    from src.config.config import ConfigurationManager
    from src.data.transformation import DataTransformation
    manager = ConfigurationManager()
    train_loader, val_loader, _ = DataTransformation(
        manager.get_data_transformation_config()
    ).get_data_loaders()
    CascadeTrainer(manager.get_cascade_config()).train(train_loader, val_loader)
