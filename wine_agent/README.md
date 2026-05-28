# Wine Intelligence Lab

This project turns `wine_quality.csv` into a small analytical product.

It trains a deep learning model to predict whether a wine is high quality, saves CSV outputs for the report, and answers simple analyst-style questions from the saved artifacts.

Quick start:

```powershell
python -m pip install -r requirements.txt

# train the model and export metrics/feature importance/predictions
python -m wine_agent.train

# ask the built-in analyst a question
python -m wine_agent.agent --question "What chemical factors most strongly predict high-quality wine?"
```

Files:

- `data.py`: loads and prepares the wine data.
- `model.py`: the PyTorch classifier.
- `train.py`: trains the model and writes CSV outputs.
- `agent.py`: answers simple analytical questions from those outputs.

Generated outputs:

- `wine_agent/outputs/wine_training_metrics.csv`
- `wine_agent/outputs/wine_test_metrics.csv`
- `wine_agent/outputs/wine_feature_importance.csv`
- `wine_agent/outputs/wine_test_predictions.csv`
- `wine_agent/outputs/wine_classifier.pth`
