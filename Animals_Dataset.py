from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Compose
import os
import pickle
import numpy as np
import cv2
from PIL import Image
from torchvision.datasets import ImageFolder

class AnimalDataset(Dataset):
    def __init__(self, root, is_train, transform=None):
        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]
        self.image_paths = []
        self.labels = []
        for index, category in enumerate(self.categories):
            subdir_path = os.path.join(data_path, category)
            for file_name in os.listdir(subdir_path):
                self.image_paths.append(os.path.join(subdir_path, file_name))
                self.labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        # image = cv2.imread(self.image_paths[item])
        image = Image.open(self.image_paths[item]).convert("RGB")
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    train_dataset = AnimalDataset(root=r"D:\AI\Dataset\animals", is_train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=0,
        shuffle=True,
        drop_last=True
    )
    for epoch in range(100):
        for images, labels in train_dataloader:
            print(images.shape, labels)
