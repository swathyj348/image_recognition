# Two Solid Projects

This repo now focuses on two stronger, self-contained projects instead of trying to do everything at once.

1. A visual AI project built around the fruit image dataset, with both image classification and reverse image search.
2. A wine-quality intelligence project that trains a deep learning classifier and answers analytical questions like which chemical factors matter most.

That gives you one vision project and one tabular analytics project, both runnable from the command line and both backed by saved outputs you can show in a portfolio or demo.

The combined notebook for the two-project walkthrough is here: [solid_projects.ipynb](solid_projects.ipynb). A small R helper is also included at [wine_agent/explore_wine.R](wine_agent/explore_wine.R).

## Project 1: Visual Search Studio

This lives in `image_agent/` and uses the provided `image_dataset/`.

What it does:
- Trains an image classifier without transformers.
- Runs inference on a single uploaded image.
- Builds a reverse image search index and returns the top 5 visually similar images.
- Exposes a simple Flask frontend for drag-and-drop style inference.

Key files:
- [image_agent/train.py](image_agent/train.py#L1-L200)
- [image_agent/infer.py](image_agent/infer.py#L1-L200)
- [image_agent/reverse_search.py](image_agent/reverse_search.py#L1-L240)
- [image_agent/cli.py](image_agent/cli.py#L1-L220)
- [image_agent/frontend.py](image_agent/frontend.py#L1-L200)

How to run:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Train the classifier
.\venv\Scripts\python.exe -m image_agent.cli train --data image_dataset --epochs 10 --batch 32 --out model_full.pth

# Build the reverse-search index
.\venv\Scripts\python.exe -m image_agent.cli build-index --data image_dataset --index image_agent/reverse_index.npz

# Search for visually similar images
.\venv\Scripts\python.exe -m image_agent.cli search --query image_dataset/Apple/some.jpg --index image_agent/reverse_index.npz --top-k 5 --out image_agent/reverse_search_results.csv

# Run the browser frontend
.\venv\Scripts\python.exe -m image_agent.frontend
```

Outputs you can expect:
- `model_full.pth` for classification.
- `image_agent/reverse_index.npz` for reverse search.
- `image_agent/reverse_search_results.csv` for the top matches.
- `image_agent/reverse_search_results_smoke.csv` for the smoke-test example.

## Project 2: Wine Intelligence Lab

This lives in `wine_agent/` and uses `wine_quality.csv`.

What it does:
- Trains a deep learning model to predict whether a wine is high quality.
- Exports training metrics and feature importance to CSV files.
- Answers analytical questions such as: “What chemical factors most strongly predict high-quality wine?”
- Produces outputs that are easy to reuse in a notebook or report.

Key files:
- [wine_agent/data.py](wine_agent/data.py#L1-L200)
- [wine_agent/model.py](wine_agent/model.py#L1-L200)
- [wine_agent/train.py](wine_agent/train.py#L1-L260)
- [wine_agent/agent.py](wine_agent/agent.py#L1-L200)

How to run:
```powershell
# Train the wine model and generate CSV outputs
.\venv\Scripts\python.exe -m wine_agent.train

# Ask the analytical agent a question
.\venv\Scripts\python.exe -m wine_agent.agent --question "What chemical factors most strongly predict high-quality wine?"
```

Outputs you can expect:
- `wine_agent/outputs/wine_training_metrics.csv`
- `wine_agent/outputs/wine_test_metrics.csv`
- `wine_agent/outputs/wine_feature_importance.csv`
- `wine_agent/outputs/wine_test_predictions.csv`
- `wine_agent/outputs/wine_classifier.pth`
- `wine_agent/outputs/r_summary.csv` from the R helper.

## Installation

The root `requirements.txt` covers both projects.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Why these two projects

These were chosen because they are the most complete and portfolio-friendly:
- The vision project shows model training, inference, a browser app, and a reverse-search retrieval system.
- The wine project shows deep learning on tabular data plus explainable analysis and CSV artifacts.

That is more credible than stretching the repo across too many half-finished tasks.

## Included Data

- `image_dataset/`
- `wine_quality.csv`
- `sentiment_analysis.csv`
- `ner.csv`
- `transfomer.txt`

Only the first two datasets are being used for the two solid projects right now.

## Notes

- The repo still contains the earlier image-classification frontend, but the new reverse-search command is part of the same visual project.
- The wine project is designed to be easy to explain in an interview or README: input data, model, outputs, and a query-answering layer.
- I can add a notebook next if you want a polished `.ipynb` walkthrough for the wine project.

Last updated: May 28, 2026
