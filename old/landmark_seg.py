import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path
from scipy.spatial import ConvexHull
from tqdm import tqdm

GT_PATH = Path('/home/phdomingues/masters/data/datasets/Occlusion')
OUTPUT_PATH = Path('/home/phdomingues/masters/results/Occlusion/landmark_seg')

def generate(shape, polygon, dst_path=None):
    mask = np.zeros([shape[0], shape[1]]) # Cria imagem com altura e largura corretas e apenas 1 canal
    mask = cv2.fillPoly(mask, [polygon], 255) # Aplica o poligono na mascara
    mask = mask.astype(np.uint8)
    if dst_path is not None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), mask) # Salva a imagem
    return mask

# Carrega o RetinaFace
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

pbar = tqdm(total=len(list(GT_PATH.glob('*'))))
for image in GT_PATH.glob('*'):
    try:
        img = cv2.imread(str(image))
        faces = app.get(img)
        try:
            hull = ConvexHull(faces[0]['landmark_2d_106'])
        except IndexError:
            pbar.write(f'No faces detected for image {image.name}')
            continue
        mask = generate(img.shape, np.array(hull.points[hull.vertices], dtype=np.int32).reshape((-1,1,2)), dst_path=OUTPUT_PATH / f'{image.stem}.jpg')
    finally:
        pbar.update(1)