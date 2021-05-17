import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from matplotlib import image as img
from skimage import io
from PIL import Image 

class CDONdataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.sublabels_mapping = {} # the sublabels have an arbitrary number, use this to enumerate them
        self.root_dir = root_dir
        self.transform = transform
        csv_path = os.path.join(self.root_dir, csv_file)
        self.annotations = pd.read_csv(csv_path, delimiter=',', header=None)

    def __len__(self):
        return int(len(self.annotations))

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "images", self.annotations.iloc[index, 0])
        with Image.open(img_path).convert('RGB') as image:
            y_wild_label = self.annotations.iloc[index, 2]
            if y_wild_label == y_wild_label:
                y_label = int(y_wild_label)
            else:
                y_label = 0
            if not y_label in self.sublabels_mapping:
                self.sublabels_mapping[y_label] = len(self.sublabels_mapping) + 1
                y_label = len(self.sublabels_mapping)
            else:
                y_label = self.sublabels_mapping[y_label]
            # y_label = torch.tensor(int(self.annotations.iloc[index, 1])) - 1 # this is the main category
            if self.transform:
                image = self.transform(image)

        return (image, torch.tensor(y_label))
