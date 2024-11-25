import cv2
import re

from itertools import tee
from insightface.app import FaceAnalysis # pip install onnxruntime insightface
from pathlib import Path
from tqdm import tqdm

from utility.file_utils import ilen

# Caminhos para os arquivos
DATASET_PATH    = Path('/home/phdomingues/experiments/data/Datasets/UNIFESP') # Caminho para as imagens
MASKS_PATH      = Path('/home/phdomingues/experiments/data/Results/Exp2-SAM/UNIFESP') # Caminho para as mascaras (usado apenas se APPLY_MASK for True)
APPLY_MASK      = False # Se True, aplica a mascara

# Carrega o RetinaFace
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
# Carrega a barra de progresso
img_paths = (f for f in DATASET_PATH.rglob('*') if f.is_file())
img_paths, _copy = tee(img_paths)
pbar = tqdm(total=ilen(_copy))

for file_path in img_paths:
    try:
        # Verifica se nao e uma pasta (se sim, nao deve ser processado)
        if not file_path.is_file():
            continue

        # Gera os caminhos para salvar a imagem da face recortada
        output_img_path = DATASET_PATH.with_name(DATASET_PATH.name+'-faces' + ('-masked' if APPLY_MASK else '')) / file_path.relative_to(DATASET_PATH)

        # Cria a pasta do novo dataset, caso necessario
        output_img_path.parent.mkdir(exist_ok=True, parents=True)

        # Verifica se o arquivo com a face recortada ja existe (se sim, pula)
        if output_img_path.is_file():
            continue

        # Carrega a imagem
        img = cv2.cvtColor(cv2.imread(str(file_path)), cv2.COLOR_BGR2RGB)

        # Detecta face
        faces = app.get(img)
        if len(faces) == 0: # Verifica se a face foi detectada
            pbar.write(f'Face nao detectada para a imagem {file_path.name}')
            continue

        # Aplica mascara, se necessario
        if APPLY_MASK:
            mask_path = tuple(MASKS_PATH.rglob(f'*{file_path.stem}*'))
            if len(mask_path) > 1:
                r1 = r"(?<=-)(0[0-9]+)" # Detecta quem tem score iniciando com 0
                r2 = r"(?<=-)([0-9]+)" # Extrai a parte decimal do score
                scores = [ # Gera uma lista dos scores de cada mascara
                    float(f'{1 if re.search(r1, str(m.stem), re.MULTILINE) else 0}.{re.search(r2, m.stem).group()}') for m in mask_path]
                best_mask_idx = scores.index(max(scores)) # Descobre o indice da melhor mascara
                best_mask = cv2.imread(str(mask_path[best_mask_idx]),0)
                _, best_mask = cv2.threshold(best_mask,127,255,cv2.THRESH_BINARY)
                best_mask = cv2.normalize(best_mask, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img = cv2.bitwise_and(img, img, mask=best_mask)
            else:
                pbar.write(f'Mascara nao encontrada para a imagem {file_path.name}.. Pulando')
                continue


        # Recorta a imagem
        ymin, xmin, ymax, xmax = map(int, faces[0]['bbox'])
        crop = img[max(xmin,0):xmax,max(ymin,0):ymax]
        
        # Salva o recorte
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) # Por algum motivo o opencv salva imagens em bgr, mesmo elas estando em formato rgb
        cv2.imwrite(str(output_img_path), crop)

    finally:
        pbar.update(1)