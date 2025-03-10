import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

import constants


class RSPBaseDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Rock, Scissors, Paper 데이터셋 로드 (기본 클래스)

        Args:
            root_dir (str): 데이터셋이 위치한 폴더
            transforms (callable, optional): Albumentations 변환 적용
        """
        self.root_dir = root_dir
        self.transforms = transforms

        self.classes = constants.CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                assert f"폴더 '{class_folder}'가 존재하지 않습니다."
            
            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                    self.image_paths.append(os.path.join(class_folder, file_name))
                    self.labels.append(self.class_to_idx[class_name])

    def set_transform(self, transforms):
        self.transforms = transforms

    def get_transform(self):
        return self.transforms

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, index):
        img_path = self.image_paths[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        return image

    def apply_transforms(self, image):
        if self.transforms:
            return self.transforms(image=image)
        else: 
            return image
        
    def to_tensor(self, image):
        image = image.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
        return torch.from_numpy(image).float() / 255.0  # normalization


class RSPTrainDataset(RSPBaseDataset):
    def __getitem__(self, index):
        image = self.load_image(index)
        image = self.apply_transforms(image)
        image = self.to_tensor(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class RSPTestDataset(RSPBaseDataset):
    def __getitem__(self, index):
        image = self.load_image(index)
        image = self.apply_transforms(image)
        image = self.to_tensor(image)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label 


class RSPInferenceDataset(RSPBaseDataset):
    def __getitem__(self, index):
        image = self.load_image(index)
        image = self.apply_transforms(image)
        image = self.to_tensor(image)
        return image
