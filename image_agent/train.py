import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from .data_loader import get_image_dataloaders


def build_model(num_classes, device):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def train(root_dir, epochs=3, batch_size=32, lr=1e-3, out_path='model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, classes = get_image_dataloaders(root_dir, batch_size=batch_size)
    model = build_model(len(classes), device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Train Epoch {epoch+1}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        print(f"Epoch {epoch+1} train loss: {running_loss/len(train_loader.dataset):.4f}")

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Val acc: {correct/total:.4f}")

    # save
    torch.save({'model_state_dict': model.state_dict(), 'classes': classes}, out_path)
    print('Model saved to', out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to image dataset root (folders per class)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', default='model.pth')
    args = parser.parse_args()
    train(args.data, epochs=args.epochs, batch_size=args.batch, out_path=args.out)
