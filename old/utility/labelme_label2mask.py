############################################################################
# THIS SCRIPT CONVERTS LABELME ANNOTATIONS OF POLYGONS TO .JPG MASK IMAGES #
############################################################################

import cv2
import json
import numpy as np
# import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

# IMGS_PATH = Path('/home/phdomingues/masters/data/UNIFESP/manual_process')
# LABELS_PATH = Path('/home/phdomingues/masters/data/UNIFESP/manual_process')
IMGS_PATH = Path(r'E:\Pedro\Faculdade\FEI-Mestrado\Datasets\UNIFESP2\to_process')
LABELS_PATH = IMGS_PATH

APPLY_MASK = True # Apply the mask to the original image
CROP = True # Crop the final image to be the exact dimensions of the mask

IMGS_EXTENSION = '.jpg'
LABELS_EXTENSION = '.json'
OUTPUT_PATH = LABELS_PATH / ('masked' if APPLY_MASK else 'non_masked')

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


label_files = list(LABELS_PATH.glob(f'*{LABELS_EXTENSION}'))
if len(label_files) == 0:
    raise FileNotFoundError('No label files found')
for label_file_path in tqdm(label_files):
    with label_file_path.open(mode='r') as label_file:
        data = json.load(label_file)
        img = cv2.imread(str(IMGS_PATH / label_file_path.with_suffix(IMGS_EXTENSION).name))
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = cv2.fillPoly(mask, # Image
                                [np.array(points, dtype=np.int32)],  # Polygon vertices
                                255, # Color
                                cv2.LINE_AA)  # Line type
            if APPLY_MASK:
                final_img = cv2.bitwise_and(img, img, mask=mask)
            else:
                final_img = img.copy()
            if CROP:
                rows, cols = np.where(np.abs(mask)>0)
                final_img = final_img[min(rows):max(rows)+1, min(cols):max(cols)+1]

            output_dir = OUTPUT_PATH / label.lower()
            output_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(output_dir / label_file_path.with_suffix('.jpg').name), final_img)
            # plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
            # plt.show()