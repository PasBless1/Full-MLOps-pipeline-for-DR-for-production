import os, sys, json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, classification_report, confusion_matrix,
)
from torch.cuda.amp import autocast
from src.models.train import HierDRNet
from src.models.cascade_model import Stage1Net, Stage2Net
from src.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig, CascadeModelConfig
from src.utils.logger import logger
from src.utils.exception import HierDRException

DR_LABELS = ["No DR", "Mild", "Moderate", "Severe", "PDR"]


class ModelEvaluator:
    def __init__(self, eval_config: ModelEvaluationConfig,
                 trainer_config: ModelTrainerConfig,
                 cascade_config: CascadeModelConfig):
        self.eval_config    = eval_config
        self.trainer_config = trainer_config
        self.cascade_config = cascade_config
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_hierdrnet(self):
        model = HierDRNet(pretrained=False, dropout=self.trainer_config.dropout).to(self.device)
        model.load_state_dict(torch.load(self.trainer_config.trained_model_path, map_location=self.device))
        model.eval()
        return model

    def _load_cascade(self):
        c = self.cascade_config
        s1 = Stage1Net(pretrained=False, dropout=c.dropout).to(self.device)
        s1.load_state_dict(torch.load(c.stage1_model_path, map_location=self.device)); s1.eval()
        s2 = Stage2Net(pretrained=False, dropout=c.dropout).to(self.device)
        s2.load_state_dict(torch.load(c.stage2_model_path, map_location=self.device)); s2.eval()
        return s1, s2

    def _eval_hierdrnet(self, model, loader):
        preds, truths = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                with autocast():
                    s1, s2, _ = model(imgs)
                s1_pred = F.softmax(s1, dim=1).argmax(dim=1).cpu()
                s2_pred = F.softmax(s2, dim=1).argmax(dim=1).cpu() + 1
                final   = torch.where(s1_pred == 0, torch.zeros_like(s2_pred), s2_pred)
                preds.extend(final.tolist()); truths.extend(labels.tolist())
        return np.array(preds), np.array(truths)

    def _eval_cascade(self, s1_model, s2_model, loader):
        preds, truths = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                with autocast():
                    s1_logits, _ = s1_model(imgs)
                s1_probs = F.softmax(s1_logits, dim=1)
                for i in range(imgs.size(0)):
                    if s1_probs[i, 1].item() <= 0.5:
                        preds.append(0)
                    else:
                        s1p = s1_probs[i, 1:2].unsqueeze(0)
                        with autocast():
                            s2_out = s2_model(imgs[i:i+1], s1p)
                        preds.append(F.softmax(s2_out, dim=1).argmax().item() + 1)
                    truths.append(labels[i].item())
        return np.array(preds), np.array(truths)

    def evaluate(self, test_loader):
        try:
            metrics = {}
            os.makedirs(self.eval_config.root_dir, exist_ok=True)

            if os.path.exists(self.trainer_config.trained_model_path):
                logger.info("Evaluating HierDRNet...")
                model = self._load_hierdrnet()
                preds, truths = self._eval_hierdrnet(model, test_loader)
                acc   = accuracy_score(truths, preds)
                kappa = cohen_kappa_score(truths, preds, weights="quadratic")
                metrics["hierdrnet"] = {"accuracy": round(acc, 4), "quadratic_kappa": round(kappa, 4)}
                logger.info(f"HierDRNet — Accuracy: {acc:.4f} | Quadratic Kappa: {kappa:.4f}")
                logger.info(classification_report(truths, preds, target_names=DR_LABELS))

            if (os.path.exists(self.cascade_config.stage1_model_path) and
                    os.path.exists(self.cascade_config.stage2_model_path)):
                logger.info("Evaluating Cascade...")
                s1, s2 = self._load_cascade()
                preds, truths = self._eval_cascade(s1, s2, test_loader)
                acc   = accuracy_score(truths, preds)
                kappa = cohen_kappa_score(truths, preds, weights="quadratic")
                metrics["cascade"] = {"accuracy": round(acc, 4), "quadratic_kappa": round(kappa, 4)}
                logger.info(f"Cascade — Accuracy: {acc:.4f} | Quadratic Kappa: {kappa:.4f}")

            os.makedirs(os.path.dirname(self.eval_config.metrics_path), exist_ok=True)
            with open(self.eval_config.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved → {self.eval_config.metrics_path}")
            return metrics
        except Exception as e:
            raise HierDRException(e, sys)


if __name__ == "__main__":
    from src.config.config import ConfigurationManager
    from src.data.transformation import DataTransformation
    manager = ConfigurationManager()
    _, _, test_loader = DataTransformation(
        manager.get_data_transformation_config()
    ).get_data_loaders()
    evaluator = ModelEvaluator(
        manager.get_model_evaluation_config(),
        manager.get_model_trainer_config(),
        manager.get_cascade_config(),
    )
    metrics = evaluator.evaluate(test_loader)
    print("\n📊 RESULTS")
    for model_name, m in metrics.items():
        print(f"  {model_name:<12} Accuracy: {m['accuracy']} | Kappa: {m['quadratic_kappa']}")
