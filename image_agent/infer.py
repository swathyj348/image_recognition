import torch
from PIL import Image
from torchvision import transforms


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    classes = ckpt.get('classes')
    return ckpt, classes


def predict(model_path, img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt, classes = load_checkpoint(model_path, device)

    # build model skeleton
    import torch.nn as nn
    from torchvision import models
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        _, pred = torch.max(out, 1)
    return classes[pred.item()]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    print(predict(args.model, args.image))
