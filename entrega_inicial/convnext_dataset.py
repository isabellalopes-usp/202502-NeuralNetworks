from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset

from PIL import Image

import numpy as np
import torch
import os

from PIL import ImageFile

# Resolve erro causado por algumas imagens do dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WildfireDataset(Dataset):
    def __init__(self, root_dir, stage, class_to_idx, idx_to_class, transform):
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage

        self.load_data()

    def load_data(self):
        self.image_paths = []
        # self.images = []
        self.labels = []

        for class_idx, class_name in self.idx_to_class.items():
            folder = os.path.join(
                self.root_dir,
                self.stage,
                class_name,
            )

            images = np.sort(os.listdir(folder))
            for name in images:
                self.image_paths.append(os.path.join(folder, name))
            self.labels += [class_idx] * len(images)

        labels, counts = np.unique(self.labels, return_counts=True)
        self.class_distribution = {k: v for k, v in zip(labels, counts)}

        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        image = np.array(Image.open(self.image_paths[idx]), dtype=np.float32)

        image = self.transform(image)
        label = self.labels[idx]
        return image, label
