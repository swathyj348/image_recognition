# Image Agent

Minimal image classification agent using PyTorch transfer learning (ResNet18).

Quick start (GPU recommended):

```bash
# install
pip install -r image_agent/requirements.txt

# train (dataset must be arranged as root/class_name/imagename.jpg)
python -m image_agent.cli train --data image_dataset --epochs 5 --batch 32 --out model.pth

# infer
python -m image_agent.cli infer --model model.pth --image image_dataset/Apple/example.jpg
```

Files:

- `data_loader.py`: builds train/val loaders from a folder of class subfolders.
- `train.py`: training loop using pretrained ResNet18.
- `infer.py`: single-image inference using saved checkpoint.
- `cli.py`: simple CLI wrapper.
