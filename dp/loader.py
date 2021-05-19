import torch
import torch.nn as nn
from torch.utils.data import Dataset

from skimage import io
from pathlib import Path
import random
import os 
import numpy as np

import random
from bs.dp.augumentation_utils import *
from imgaug import augmenters as iaa

class ImageDataset(Dataset):
    def __init__(self, data_dir:str, fold:str, resize_size:int):
        super(ImageDataset, self).__init__()
        self.data_dir = data_dir
        
        self.image_dir = Path(data_dir)/fold
        self.image_files = [str(path) for path in Path(self.image_dir).glob('*/*')]
        
        random.shuffle(self.image_files)

        self.fold = fold

        self.resize_size = resize_size

        self.mapping = {}

    def __len__(self):
        return len(self.image_files)

    @property
    def num_classes(self):
        return len(self.mapping.keys())

    
    def __getitem__(self, idx):
        image = None

        image_path = self.image_files[idx]
        image_id = image_path.split('/')[-1].replace('.png', '')
        image = io.imread(image_path)
        image = cv2.resize(image[..., :3], (self.resize_size, self.resize_size), interpolation=cv2.INTER_NEAREST)

    
        #label
        parts = image_path.split('/')
        label = parts[-2]
        label = self.mapping[label]


        if self.fold == 'train':
            image = self.augument(image)

        
        image = self.normalize(image)
        image = torch.from_numpy(image.transpose((2, 0, 1)).copy()).float()

        return {'image': image, 'label': label, 'image_id': image_id}

    def augument(self, image):
        k = random.randrange(4)
        image = np.rot90(image, k=k)

        if random.random() > 0.5:
            image = image[::-1, ...]

        if random.random() > 0.5:
            image = image[:, ::-1, ...]


        if random.random() > 0.95:
            image = saturation(image, 0.9 + random.random() * 0.2)

        elif random.random() > 0.95:
            image = brightness(image, 0.9 + random.random() * 0.2)
            
        elif random.random() > 0.95:
            image = contrast(image, 0.9 + random.random() * 0.2)

        return image 


    def normalize(self, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        image = np.asarray(image, dtype='float32') / 255
        for i in range(3):
            image[..., i] = (image[..., i] - mean[i]) / std[i]

        return image




