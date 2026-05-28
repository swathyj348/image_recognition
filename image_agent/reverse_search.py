from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import models


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _build_extractor():
    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
    return extractor.eval(), weights.transforms()


def _list_images(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    return sorted(
        path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _embed_image(path: Path, model: torch.nn.Module, preprocess, device: torch.device) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).flatten().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def build_index(
    image_root: str | Path,
    index_path: str | Path = 'image_agent/reverse_index.npz',
    max_images: int | None = None,
) -> Path:
    image_paths = _list_images(image_root)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise ValueError(f'No images found in {image_root}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = _build_extractor()
    model = model.to(device)

    embeddings = []
    classes = []
    paths = []
    for image_path in image_paths:
        embeddings.append(_embed_image(image_path, model, preprocess, device))
        classes.append(image_path.parent.name)
        paths.append(str(image_path))

    payload = {
        'embeddings': np.stack(embeddings),
        'paths': np.array(paths),
        'classes': np.array(classes),
    }
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(index_path, **payload)
    return index_path


def load_index(index_path: str | Path):
    data = np.load(index_path, allow_pickle=True)
    return data['embeddings'], data['paths'], data['classes']


def search_similar_images(
    query_image: str | Path,
    index_path: str | Path = 'image_agent/reverse_index.npz',
    top_k: int = 5,
):
    embeddings, paths, classes = load_index(index_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = _build_extractor()
    model = model.to(device)

    query_embedding = _embed_image(Path(query_image), model, preprocess, device)
    scores = embeddings @ query_embedding
    order = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in order:
        results.append(
            {
                'path': str(paths[idx]),
                'class': str(classes[idx]),
                'similarity': float(scores[idx]),
            }
        )
    return results


def results_to_csv(results: Iterable[dict], csv_path: str | Path) -> Path:
    import csv

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'class', 'similarity'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return csv_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Reverse image search using ResNet18 embeddings')
    sub = parser.add_subparsers(dest='cmd')

    build_parser = sub.add_parser('build-index')
    build_parser.add_argument('--data', required=True)
    build_parser.add_argument('--index', default='image_agent/reverse_index.npz')
    build_parser.add_argument('--limit', type=int, default=None)

    search_parser = sub.add_parser('search')
    search_parser.add_argument('--query', required=True)
    search_parser.add_argument('--index', default='image_agent/reverse_index.npz')
    search_parser.add_argument('--top-k', type=int, default=5)
    search_parser.add_argument('--out', default='image_agent/reverse_search_results.csv')

    args = parser.parse_args()
    if args.cmd == 'build-index':
        out = build_index(args.data, args.index, args.limit)
        print(f'Index saved to {out}')
    elif args.cmd == 'search':
        results = search_similar_images(args.query, args.index, args.top_k)
        results_to_csv(results, args.out)
        for row in results:
            print(row)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
