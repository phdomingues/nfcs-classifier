import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utility.file_utils import count_files

MASKS_PATH = Path('/home/phdomingues/masters/results/Occlusion/sam') # O nome da pasta sera usado como referencia para salvar os scores
GT_PATH = Path('/home/phdomingues/masters/data/datasets/Occlusion/masks')
OUTPUT_FILE = Path('/home/phdomingues/masters/results/Occlusion/dice_scores.csv')

COL_NAME = MASKS_PATH.name

images = []
scores = {COL_NAME: []}
pbar = tqdm(total=count_files(MASKS_PATH, recursive=True, pattern='*'))
if pbar.total == 0:
    exit(0)
for image in MASKS_PATH.glob('*'):
    try:
        mask = cv2.imread(str(image), 0)
        _, mask = cv2.threshold(mask,127,1,0)
        try:
            gt = cv2.imread(str(next(GT_PATH.glob(f'{image.stem}*'))), 0)
            _, gt = cv2.threshold(gt,127,1,0)
        except StopIteration:
            pbar.write(f"No ground truth found for {image}")
            continue
        dice = np.sum(mask[gt==1])*2.0 / (np.sum(mask) + np.sum(gt))
        images.append(image.stem)
        scores[COL_NAME].append(dice)
    finally:
        pbar.update(1)

df = pd.DataFrame(scores, index=images)

if OUTPUT_FILE.is_file():
    df = pd.concat([pd.read_csv(OUTPUT_FILE, index_col=0), df], axis=1)

df.to_csv(OUTPUT_FILE)