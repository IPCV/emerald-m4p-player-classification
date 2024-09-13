import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_patch(frame, bbox, copy=True):
    x1, y1, x2, y2 = bbox
    patch = frame[y1:y2, x1:x2, ...]
    return np.copy(patch) if copy else patch


class PatchesDataset(Dataset):
    def __init__(self, images, detections):
        self.patches = {}
        self.detections = detections
        for i, img in enumerate(images):
            self.patches[i] = [get_patch(img, bb) for bb in detections[i]]

        self.patch_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patches = torch.stack([self.patch_transform(p) for p in self.patches[idx]])
        return patches, self.detections[idx]
