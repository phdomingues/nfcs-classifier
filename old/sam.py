import logging
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from insightface.app import FaceAnalysis

from segment_anything import SamPredictor, sam_model_registry

INPUT_DATASET = Path('/home/phdomingues/masters/data/datasets/Occlusion')
RESULTS_PATH = Path('/home/phdomingues/masters/results/Occlusion/sam')
MODEL_PATH = Path('/home/phdomingues/experiments/data/Models/sam_vit_h_4b8939.pth')

# ====== SETUP LOG HANDLER ====== #
def setup_logger(log_level, pbar) -> None:
    Path('Logs').mkdir(exist_ok=True, parents=True)
    # Create logger
    logger = logging.getLogger('Mosaic-Pipeline')
    # Setup output formating
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    # Setup log lovel
    logger.setLevel(logging.getLevelName(log_level))
    # Setup file handler
    file_name = datetime.now().strftime("%Y%m%dT%H%M%S-Mosaic")
    file_handler = logging.FileHandler("{0}/{1}.log".format('Logs', file_name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Setup stream handler
    #stream_handler = logging.StreamHandler(sys.stdout)
    #stream_handler.setFormatter(formatter)
    #logger.addHandler(stream_handler)
    return logger


# Prepara os caminhos
images_path = INPUT_DATASET
results_path = RESULTS_PATH
# Prepara variaveis de execucao
save_all = False
threshold = 3 if save_all else 1
# Carrega o RetinaFace
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
# Carrega o SAM
checkpoint_path = MODEL_PATH
checkpoint = sam_model_registry['vit_h'](checkpoint=checkpoint_path)
checkpoint.to(device='cuda')
sam = SamPredictor(checkpoint)

# Iterando no dataset
with logging_redirect_tqdm():
    image_pbar = tqdm(desc='Imagens processadas', leave=True)
    logger = setup_logger('INFO', image_pbar)
    for image in images_path.rglob('*'):
        if not image.is_file():
            image_pbar.update(1)
            logger.warning(f'{image.name} is a directory.. Skipping')
            continue
        if len(list(results_path.glob(f'{image.stem}*'))) == threshold:
            image_pbar.update(1)
            logger.warning(f'Image {image.name} already processed.. Skipping')
            continue
        # Carrega a imagem
        img = cv2.imread(str(image))
        if img is None:
            logger.warning(f'Error reading file {image.name}... Skipping')
            image_pbar.update(1)
            continue
        # Detecta a face
        faces = app.get(img)
        if len(faces) < 1:
            logger.warning(f'Skipping image {image.name}... no faces found')
            image_pbar.update(1)
            continue
        # Segmenta (SAM)
        sam.set_image(img)
        masks, scores, logits = sam.predict(box=faces[0]['bbox'])
        # Salva a(s) mascaras geradas dependendo da config
        best_score = -1
        best_image = None
        best_mask = None
        for mask, score in zip(masks, scores):
            # Converte a mascara de bool para unsigned int 8 bits (0-255)
            mask = mask.astype(np.uint8)
            mask[mask>0] = 255 # Converte as mascaras de 0/1 para 0/255, apenas para facilitar visualização 
            # Prepara o caminho para salvar a mascara resultante
            results_path.mkdir(parents=True, exist_ok=True) # Cria a pasta se necessário
            score_str = str(score).replace('.','-')
            # Verifica por novo melhor
            if score > best_score:
                best_image = image
                best_mask = mask
                best_score = score
            # Salva resultados
            if save_all:
                cv2.imwrite(str(results_path / f"{image.stem}_{score_str}.png"), mask)
        if not save_all:
            best_score_str = str(best_score).replace('.','-')
            cv2.imwrite(str(results_path / f"{best_image.stem}.png"), best_mask)
        image_pbar.update(1)