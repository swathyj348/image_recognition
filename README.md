# Projects

1. Visual image classification and reverse image search
2. Wine quality analysis with feature importance

Walkthrough: [solid_projects.ipynb](solid_projects.ipynb)

## Image Classification

Located in `image_agent/` with `image_dataset/`.

Files:
- [image_agent/train.py](image_agent/train.py)
- [image_agent/infer.py](image_agent/infer.py)
- [image_agent/reverse_search.py](image_agent/reverse_search.py)
- [image_agent/cli.py](image_agent/cli.py)
- [image_agent/frontend.py](image_agent/frontend.py)

Setup:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run:
```powershell
# Train classifier
.\venv\Scripts\python.exe -m image_agent.cli train --data image_dataset --epochs 10 --batch 32 --out model_full.pth

# Build reverse-search index
.\venv\Scripts\python.exe -m image_agent.cli build-index --data image_dataset --index image_agent/reverse_index.npz

# Search for similar images
.\venv\Scripts\python.exe -m image_agent.cli search --query image_dataset/Apple/some.jpg --index image_agent/reverse_index.npz --top-k 5 --out image_agent/reverse_search_results.csv

# Browser frontend
.\venv\Scripts\python.exe -m image_agent.frontend
```

Outputs:
- `model_full.pth` — classification model
- `image_agent/reverse_index.npz` — reverse search index
- `image_agent/reverse_search_results.csv` — top similar images

## Wine Analysis

Located in `wine_agent/` with `wine_quality.csv`.

Files:
- [wine_agent/data.py](wine_agent/data.py)
- [wine_agent/model.py](wine_agent/model.py)
- [wine_agent/train.py](wine_agent/train.py)
- [wine_agent/agent.py](wine_agent/agent.py)

Run:
```powershell
# Train model
.\venv\Scripts\python.exe -m wine_agent.train

# Ask analytical questions
.\venv\Scripts\python.exe -m wine_agent.agent --question "What chemical factors most strongly predict high-quality wine?"
```

Outputs:
- `wine_agent/outputs/wine_training_metrics.csv`
- `wine_agent/outputs/wine_test_metrics.csv`
- `wine_agent/outputs/wine_feature_importance.csv`
- `wine_agent/outputs/wine_test_predictions.csv`
- `wine_agent/outputs/wine_classifier.pth`
