# Wine Analysis

Trains a deep learning model on `wine_quality.csv` to predict high-quality wines. Exports metrics and feature importance.

Run:
```powershell
python -m pip install -r requirements.txt
python -m wine_agent.train
python -m wine_agent.agent --question "What chemical factors most strongly predict high-quality wine?"
```

Files:
- `data.py` — loads and prepares wine data
- `model.py` — PyTorch classifier
- `train.py` — model training and CSV outputs
- `agent.py` — analytical queries

Outputs:
- `outputs/wine_training_metrics.csv`
- `outputs/wine_test_metrics.csv`
- `outputs/wine_feature_importance.csv`
- `outputs/wine_test_predictions.csv`
- `outputs/wine_classifier.pth`
