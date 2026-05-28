# Visual Search Studio

Minimal vision project using PyTorch transfer learning (ResNet18).

It includes image classification plus reverse image search over the same dataset.

Quick start (GPU recommended):

```bash
# install
pip install -r image_agent/requirements.txt

# train (dataset must be arranged as root/class_name/imagename.jpg)
python -m image_agent.cli train --data image_dataset --epochs 5 --batch 32 --out model.pth

# infer
python -m image_agent.cli infer --model model.pth --image image_dataset/Apple/example.jpg

# reverse image search index
python -m image_agent.cli build-index --data image_dataset --index image_agent/reverse_index.npz

# search for the top 5 similar images
python -m image_agent.cli search --query image_dataset/Apple/example.jpg --index image_agent/reverse_index.npz --top-k 5 --out image_agent/reverse_search_results.csv
```

Files:

- `data_loader.py`: builds train/val loaders from a folder of class subfolders.
- `train.py`: training loop using pretrained ResNet18.
- `infer.py`: single-image inference using saved checkpoint.
- `reverse_search.py`: builds a reverse-search index and returns the top matches.
- `cli.py`: simple CLI wrapper.
