from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_wine_frame, prepare_wine_data, split_and_scale
from .model import WineClassifier


def _tensor_loader(features, target, batch_size=64, shuffle=False):
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(target, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate(model, features, target):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(target, predictions)),
        'roc_auc': float(roc_auc_score(target, probabilities)),
        'average_precision': float(average_precision_score(target, probabilities)),
    }


def permutation_importance(model, features, target, feature_names, repeat_seed=42):
    baseline = _evaluate(model, features, target)['accuracy']
    rng = np.random.default_rng(repeat_seed)
    importance_rows = []

    for column_index, feature_name in enumerate(feature_names):
        perturbed = features.copy()
        rng.shuffle(perturbed[:, column_index])
        score = _evaluate(model, perturbed, target)['accuracy']
        importance_rows.append(
            {
                'feature': feature_name,
                'baseline_accuracy': baseline,
                'permuted_accuracy': score,
                'importance_drop': baseline - score,
            }
        )

    importance_frame = pd.DataFrame(importance_rows).sort_values('importance_drop', ascending=False)
    return importance_frame


def train_wine_model(
    csv_path: str | Path = 'wine_quality.csv',
    output_dir: str | Path = 'wine_agent/outputs',
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_wine_frame(csv_path)
    features, target, feature_names, feature_frame = prepare_wine_data(frame)
    x_train, x_val, x_test, y_train, y_val, y_test, scaler = split_and_scale(features, target)

    train_loader = _tensor_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    model = WineClassifier(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    metrics_rows = []
    best_state = None
    best_val_auc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = _evaluate(model, x_val, y_val)
        metrics_rows.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_roc_auc': val_metrics['roc_auc'],
                'val_average_precision': val_metrics['average_precision'],
            }
        )

        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_state = {
                'model_state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'feature_names': feature_names,
            }

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / 'wine_training_metrics.csv'
    metrics_frame.to_csv(metrics_path, index=False)

    test_metrics = _evaluate(model, x_test, y_test)
    test_frame = pd.DataFrame([test_metrics])
    test_frame.to_csv(output_dir / 'wine_test_metrics.csv', index=False)

    importance_frame = permutation_importance(model, x_val, y_val, feature_names)
    importance_path = output_dir / 'wine_feature_importance.csv'
    importance_frame.to_csv(importance_path, index=False)

    sample_predictions = pd.DataFrame(
        {
            'actual_high_quality': y_test.astype(int),
            'predicted_probability': torch.sigmoid(
                model(torch.tensor(x_test, dtype=torch.float32))
            ).detach().cpu().numpy(),
        }
    )
    sample_predictions['predicted_label'] = (sample_predictions['predicted_probability'] >= 0.5).astype(int)
    sample_predictions.to_csv(output_dir / 'wine_test_predictions.csv', index=False)

    checkpoint_path = output_dir / 'wine_classifier.pth'
    torch.save(best_state, checkpoint_path)
    return {
        'checkpoint_path': str(checkpoint_path),
        'metrics_path': str(metrics_path),
        'importance_path': str(importance_path),
        'test_metrics': test_metrics,
    }


if __name__ == '__main__':
    result = train_wine_model()
    print(result)
