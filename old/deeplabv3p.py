from mmseg.apis import inference_segmentor, init_segmentor
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np

# Paths
config_file = '/home/phdomingues/masters/libs/face-occlusion-generation/dataset/Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets/weights/deeplabv3plus_celebA_train_wo_natocc_wsot/deeplabv3plus_celebA_train_wo_natocc_wsot.py'
checkpoint_file = '/home/phdomingues/masters/libs/face-occlusion-generation/dataset/Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets/weights/deeplabv3plus_celebA_train_wo_natocc_wsot/iter_27600.pth'
result_path =  Path('/home/phdomingues/masters/results')
dataset_path = Path('/home/phdomingues/masters/data/datasets/Occlusion')

result_path = result_path / 'Occlusion' / 'deeplabv3p'
result_path.mkdir(parents=True, exist_ok=True)

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

for img in tqdm(dataset_path.glob('*'), total=len(list(dataset_path.glob('*')))):
    # test a single image and show the results
    # img = 'D:/ComputerScience/Mestrado/data/Exemplos_sem_oclusao/com_dor/07_114907.bmp'  # or img = mmcv.imread(img), which will only load it once
    result = inference_segmentor(model, img)

    # visualize the results in a new window
    #model.show_result(img, result, show=True)

    r = result[0]
    r[r==1] = 255
    cv2.imwrite(str(result_path / img.with_suffix('.jpg').name), r)
    
    #print(np.unique(cv2.imread(str(result_path / img.with_suffix('.jpg').name), 0)))
    # with (result_path/'result.npy').open('wb') as f:
    #     np.save(f, result[0])


    # Save the image-mask overlay
    # you can change the opacity of the painted segmentation map in (0, 1].
    #model.show_result(img, result, out_file=result_path / f'{img.stem}.jpg', opacity=0.5)