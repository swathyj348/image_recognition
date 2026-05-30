# Image Classification

Vision project using PyTorch transfer learning (ResNet18) for image classification and reverse image search.

Run:
```powershell
pip install -r image_agent/requirements.txt

# Train
python -m image_agent.cli train --data image_dataset --epochs 5 --batch 32 --out model.pth

# Infer
python -m image_agent.cli infer --model model.pth --image image_dataset/Apple/example.jpg

# Reverse image search index
python -m image_agent.cli build-index --data image_dataset --index image_agent/reverse_index.npz

# Search for similar images
python -m image_agent.cli search --query image_dataset/Apple/example.jpg --index image_agent/reverse_index.npz --top-k 5 --out image_agent/reverse_search_results.csv
```

Files:
- `data_loader.py` — train/val loaders from class subfolders
- `train.py` — training loop using ResNet18
- `infer.py` — single-image inference
- `reverse_search.py` — reverse-search index and top matches
- `cli.py` — CLI wrapper
