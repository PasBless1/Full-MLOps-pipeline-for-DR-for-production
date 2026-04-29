# HierDR-Net MLOps Pipeline

Two-stage hierarchical EfficientNet-B4 for Diabetic Retinopathy Detection.

## Quick Start (Colab)

```bash
pip install -r requirements.txt
pip install -e .
```

## Run Individual Scripts

```bash
python src/data/ingestion.py
python src/data/validation.py
python src/data/transformation.py
python src/models/train.py
python src/models/cascade_trainer.py
python src/models/evaluate.py
```

## Run Full Pipelines

```bash
# HierDRNet end-to-end
python src/pipelines/training_pipeline.py

# Cascade end-to-end
python src/pipelines/cascade_training_pipeline.py

# Evaluate both models
python src/pipelines/evaluation_pipeline.py
```

## API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8080
# → http://localhost:8080/docs
```

## Pipeline Architecture

```
Data Ingestion → Data Validation → Data Transformation
       ↓
   HierDRNet (shared backbone, masked hierarchical loss)
       ↓
   Cascade (Stage1Net binary → Stage2Net severity)
       ↓
   Evaluation (Accuracy + Quadratic Cohen's Kappa)
       ↓
   FastAPI /predict endpoint
```

## Key Hyperparameters (params.yaml)

| Param | Value |
|---|---|
| IMAGE_SIZE | 260 |
| BATCH_SIZE | 8 |
| EPOCHS | 50 |
| LEARNING_RATE | 0.0001 |
| EARLY_STOPPING_PATIENCE | 5 |
