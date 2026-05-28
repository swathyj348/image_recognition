from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_image_dataloaders(root_dir, img_size=224, batch_size=32, val_split=0.2, num_workers=4):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform_train)
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # ensure val uses validation transforms
    val_ds.dataset = datasets.ImageFolder(root=root_dir, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = full_dataset.classes
    return train_loader, val_loader, classes
