import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CIFAR10(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        self.data = []
        self.classes = sorted(os.listdir(os.path.join(root_dir, 'train' if train else 'test')))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self._load_data()

    def _load_data(self):
        data_dir = os.path.join(self.root_dir, 'train' if self.train else 'test')
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.data.append((img_path, class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target


    def print_class_distribution(self):
        class_count = {class_name: 0 for class_name in self.classes}
        for _, target in self.data:
            class_count[self.classes[target]] += 1

        for class_name, count in class_count.items():
            print(f"{class_name}: {count} images")


if __name__ == "__main__":
    # Example usage:
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_dir = '../data/CIFAR-10-images'

    train_dataset = CIFAR10(root_dir=data_dir, train=True, transform=transform)
    test_dataset = CIFAR10(root_dir=data_dir, train=False, transform=transform)

    train_dataset.print_class_distribution()
    test_dataset.print_class_distribution()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)