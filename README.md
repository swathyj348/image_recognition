# Image Agent Project

This repository contains a small, runnable image-classification agent built with PyTorch and a minimal Flask frontend for inference. It's designed as an MVP so you can train a model on your `image_dataset/` and run predictions both from the command line and via a browser UI.

If anything is unclear or you'd like me to add more features (API endpoints, prettier UI, Dockerfile, CI), tell me which and I'll implement it.

**What's Included**
- **Dataset (provided):** `image_dataset/` — one folder per class (Apple, Banana, mango, etc.).
- **Agent package:** [image_agent/train.py](image_agent/train.py#L1-L200), [image_agent/infer.py](image_agent/infer.py#L1-L200), [image_agent/data_loader.py](image_agent/data_loader.py#L1-L200), [image_agent/cli.py](image_agent/cli.py#L1-L200)
- **Frontend:** [image_agent/frontend.py](image_agent/frontend.py#L1-L200) and template [image_agent/templates/index.html](image_agent/templates/index.html#L1-L200)
- **Requirements:** [image_agent/requirements.txt](image_agent/requirements.txt#L1-L50)
- **Saved models (examples):** `image_agent_quick.pth`, `model_full.pth` (generated after training)

**High-level Flow**
1. Prepare the dataset: place images under `image_dataset/<class_name>/*.jpg`.
2. Create and activate a Python virtual environment.
3. Install dependencies from `image_agent/requirements.txt`.
4. Train a model with the CLI or Python API — outputs a checkpoint (`.pth`).
5. Run inference from the CLI or via the Flask frontend.

**Create and activate the venv (Windows PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Install dependencies**
```powershell
python -m pip install --upgrade pip
python -m pip install -r image_agent/requirements.txt
```

**Training (CLI)**
Train quickly for testing or run a longer job:
```powershell
# Quick test (1 epoch)
.\venv\Scripts\python.exe -m image_agent.cli train --data image_dataset --epochs 1 --batch 16 --out image_agent_quick.pth

# Full training (recommended on GPU if available)
.\venv\Scripts\python.exe -m image_agent.cli train --data image_dataset --epochs 10 --batch 32 --out model_full.pth
```

- The training script uses a pretrained ResNet18 and fine-tunes the final linear layer. It auto-detects GPU (`cuda`) if available.
- Checkpoint format: a dict with `model_state_dict` and `classes`.

**Inference (CLI)**
```powershell
.\venv\Scripts\python.exe -m image_agent.cli infer --model model_full.pth --image image_dataset/Apple/some.jpg
```

**Run the Flask frontend**
The frontend accepts uploads and shows the predicted class. Uploaded images are stored in `image_agent/static/uploads/` so the app can serve them.
```powershell
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m image_agent.frontend
# Then open http://localhost:5000 in your browser
```

**Files of interest**
- [image_agent/data_loader.py](image_agent/data_loader.py#L1-L200): builds train/val DataLoaders from `ImageFolder`.
- [image_agent/train.py](image_agent/train.py#L1-L200): training loop (ResNet18), saves checkpoint.
- [image_agent/infer.py](image_agent/infer.py#L1-L200): loads checkpoint and runs single-image prediction.
- [image_agent/cli.py](image_agent/cli.py#L1-L200): convenience CLI wrapper for train/infer.
- [image_agent/frontend.py](image_agent/frontend.py#L1-L200): Flask app exposing a simple upload form.
- [image_agent/requirements.txt](image_agent/requirements.txt#L1-L50): dependencies (PyTorch, torchvision, Flask, etc.).

**Notes & tips**
- GPU: If you have a CUDA GPU and a matching PyTorch build, training will be much faster. The scripts use `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` automatically.
- Data size: If your dataset is large, increase `num_workers` in `get_image_dataloaders()` (see [image_agent/data_loader.py](image_agent/data_loader.py#L1-L200)).
- Checkpoint reuse: `infer.py` expects the saved `classes` list inside the checkpoint — don't strip it.
- Pretrained weights: torchvision warns about `pretrained` being deprecated; that is cosmetic. You can replace with the `weights=` enum if you prefer.

**Troubleshooting**
- "ModuleNotFoundError: No module named 'torch'" — make sure you run Python from the activated venv where you installed packages.
- If training stalls or downloads weights slowly, ensure you have internet access (first run downloads ResNet weights) or pre-download weights to the cache.
- Windows venv creation: if `python -m venv venv` errors about copying venvlauncher, try running PowerShell as Administrator or use an existing conda environment.

**Suggested next steps I can implement**
- Add API endpoints: `POST /predict` returning JSON (useful for programmatic clients). 
- Dockerfile + docker-compose so you can run the frontend and model in a container.
- Add NER / sentiment / wine-quality small agents to the same interface.
- Add unit tests and a small evaluation script to produce confusion matrices and class-wise metrics.

If you want any of the next steps, tell me which one and I’ll add it. If you'd like, I can also commit these changes and create a small Git branch for you.

---
Last updated: May 28, 2026
