from pathlib import Path

import pandas as pd
import torch
import numpy as np

from PIL import Image

from math import ceil
from torch.utils.data import Dataset as Dataset_torch
from torchvision.transforms.functional import (resize, pad)

from projeto_dor import InfantDataset, NFCS

# IMAGES_PATH = Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\results\mosaic-2\ComArtefatoCompleto\masked')
# CSV_PATH = Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\src\Tatiany-Regioes\data\occlusion\Avaliadores_aparato-consenso-presente.csv')
# FILE_EXTENSION = '.jpg'
IMAGES_PATH = Path(r'D:\ComputerScience\Mestrado\results\mosaic-2\iCOPE')
CSV_PATH = Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\src\Tatiany-Regioes\data\icope\icope_labels.csv')
MASKED = True


class InfantPainDataset(Dataset_torch):
    """Base dataset to load images and labels from the csv file + resize/pad to 224x224"""
    def __init__(self, image_dir:str, csv_path:str, label_col:str) -> None:
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.label_col = label_col
        self.image_dir = Path(image_dir)
        for _ in self:
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        label = row[self.label_col]

        # Search image file
        try:
            img_path = next(self.image_dir.glob(f"{row['image_id']}.*"))
        except StopIteration:
            print(f"File {self.image_dir / row['image_id']} not found... dropping from dataset")
            self.data.drop(self.data[self.data['image_id']==row['image_id']].index, axis=0, inplace=True)
            return None, None, None
            # raise FileNotFoundError(f"No file named {row['image_id']} found at {self.image_dir}")
        # Load image
        img = Image.open(img_path)
        # Resize keeping aspect ratio
        img = self.resize_pad(img, 224)  # ViT resizes to 224x224, so we resize first to avoid distortions

        return img, label, img_path

    def resize_pad(self, img, smaller_side_size:int):
        # img = img.rotate(90, expand=1) # Rotation for test purposes
        # Resize
        ar = img.height/img.width # Aspect Ratio
        img = resize(img, (smaller_side_size, round(smaller_side_size/ar)) if ar > 1 else (round(smaller_side_size*ar), smaller_side_size)) # Resize to fit the smaller dimension to smaller_side_size
        # Padding
        pad_horizontal = (smaller_side_size-img.width)/2 # How much to pad horizontally
        pad_vertical = (smaller_side_size-img.height)/2 # Hor much to pad vertically
        img = pad(img, padding=(ceil(pad_horizontal), ceil(pad_vertical), int(pad_horizontal), int(pad_vertical)), padding_mode ='constant') # (left, top, right, bottom)
        return img

def transforms(batch, processor, augmentations):
    """Apply augmentations and necessary preprocessing using the processor instance"""
    inputs = processor([augmentations(x) for x in batch['image']], return_tensors='pt'),
    # For some reason, some times the output of the line above is a tuple and other times is a dictionary.. so the logic bellow is necessary
    if isinstance(inputs, tuple):
        if len(inputs) > 1:
            raise Exception(f"Unexpected len for input in transforms ({len(inputs)})")
        inputs = inputs[0]
    else:
        raise Exception("Unexpected len for input during transforms...")
    inputs['labels'] = batch['label']
    inputs['files'] = batch['image_file']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def compute_metrics(p, metrics):
    pred = np.argmax(p.predictions, axis=1)
    val = [metric.compute(predictions=pred, references=p.label_ids) for metric in metrics]
    return {k: v for d in val for k, v in d.items()} # Merge list of metrics into a single dictionary with all metrics

ds = InfantDataset(
    image_dir=IMAGES_PATH / 'masked' if MASKED else 'non_masked',
    csv_path=CSV_PATH,
    nfcs_component=NFCS.BA,
)

for i in ds:
    print(i)