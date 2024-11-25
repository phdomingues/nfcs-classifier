import json

import numpy as np
from datasets import load_metric, Dataset
from functools import partial
from pathlib import Path
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torch import nn
from torchvision.transforms import (
    Compose,
    ColorJitter,
    GaussianBlur,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomAffine)
from tqdm import tqdm
from typing import List

from projeto_dor import *

class ClassWeightedTrainer(Trainer):
    def __init__(self, class_w, *args, **kwargs):
        self.class_w = class_w
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_w.tolist(), device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def gen(ds=[]):
    """Function to convert InfantDataset to generator"""
    for img, label, img_path in ds:
        yield {
            'image': img,
            'label': label,
            'image_file': img_path.name
            }

full_dataset = DatasetMeta(
    name='Full', csv_path=r"D:\ComputerScience\Mestrado\data\UNIFESP\NEW_GT.csv"
)
unifesp_dataset = DatasetMeta(
    name='UNIFESP', csv_path=r"D:\ComputerScience\Mestrado\data\UNIFESP\NEW_GT_UNIFESP.csv"
)

TRAINED_MOODELS_PATH = Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\results\tests')
DATA_PATH = Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\results\Optuna\ViT')
masked_images_path = r"D:\ComputerScience\Mestrado\data\masked_crops"
non_masked_images_path = r"D:\ComputerScience\Mestrado\data\non_masked_crops"

# Output control
df_data = {}

for file in tqdm(list(DATA_PATH.rglob('summary.json')), desc="Model configs processed"):
    with file.open('r') as f:
        metadata = json.load(f)

    # Model
    DATASET:DatasetMeta   = unifesp_dataset if metadata['dataset'] == unifesp_dataset.name else full_dataset
    NFCS_REGION:NFCS      = NFCS[metadata['nfcs']]
    MASKED:bool           = metadata['masked'] # If true use the masked crops if false use the cropped only
    VALIDATION_SIZE:float = metadata['val_split'] # % of images used for validation
    METRICS:List[str]     = ["accuracy", "f1"] # For a list of available metrics call the function list_metrics()
    OBJECTIVE_METRIC      = "eval_f1" # Metric that will be optimized by optuna
    GREATER_IS_BETTER     = True # Relative to the objective metric
    # Model path to download from huggingface
    model_name_or_path:str = metadata['model']
    # Dataset path
    csv_path = DATASET.csv_path

    is_frozen = "frozen" in metadata["description"].lower()
    is_weighted = "non weighted loss" not in metadata["description"].lower()

    model_id = \
        f'{DATASET.name}-' \
        f'{"" if MASKED else "non_"}masked-' \
        f'{"non_" if not is_weighted else ""}weighted-' \
        f'{"non_" if not is_frozen else ""}frozen'

    # Processor (pre-process the images to fit the expected input)
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    if df_data.get(model_id) is None:
        df_data[model_id] = dict()

    saved_model_path = TRAINED_MOODELS_PATH / f'{model_id}-{NFCS_REGION.name}'
    if saved_model_path.exists():
        best_model = ViTForImageClassification.from_pretrained(
            next((TRAINED_MOODELS_PATH / f'{model_id}-{NFCS_REGION.name}').glob('checkpoint*')),
            local_files_only=True,
        )
        best_model.cuda()
    else:

        output_dir = Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\results\optimized_ViT_models', model_id+f'-{NFCS_REGION.name}')
        output_dir.mkdir(exist_ok=True, parents=True)

        # Image augmentations
        augmentations = Compose([ # https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html
            RandomAffine(degrees=(-45,45), translate=(0.1, 0.3), scale=(0.7, 0.9)),
            ColorJitter(brightness=.4, hue=0.3, saturation=0.6), # not contrast, or the background will be afected too
            GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
            RandomHorizontalFlip(p=0.15),
            RandomVerticalFlip(p=0.15),
        ])

        # Model
        def model_init(trial):
            # Model with pre-loaded weights and + an untrained classifier head
            model = ViTForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=2,
                # id2label={str(i): c for i, c in enumerate(labels)},
                # label2id={c: str(i) for i, c in enumerate(labels)}
            )

            #### ====== FREEZE LAYERS ====== ####
            if is_frozen:
                # Freeze parameters for the base model (don't freeze the classification head)
                for param in model.base_model.parameters():
                    param.requires_grad = False
            #####################################

            return model

        # Dataset
        torch_ds = InfantDataset(
            image_dir=masked_images_path if MASKED else non_masked_images_path,
            csv_path=csv_path,
            nfcs_component=NFCS_REGION
        )
        # Cast dataset from pytorch to huggingface
        ds = Dataset.from_generator(gen, gen_kwargs=({'ds': torch_ds}))
        # Set train test split
        # The train and test are beeing reversed and shuffle is set to False so the UNIFESP part of the full dataset is the only one used for validation 
        # (since it is loaded first and the train_test_split sets the first images as training by default)
        ds = ds.train_test_split(test_size=1-VALIDATION_SIZE, shuffle=False)
        ds['train'], ds['test'] = ds['test'], ds['train']
        # Add augmentations and pre-process
        ds['train'].set_transform(partial(transforms, processor=processor, augmentations=augmentations))
        ds['test'].set_transform(partial(transforms, processor=processor, augmentations=Compose([])))  # No augmentations in the test set

        total = ds['train'].num_rows
        class_w = np.zeros(2)
        for data in ds['train']:
            class_w[data['labels']] += 1
        class_w /= total


        # HuggingFace trainer
        training_args = TrainingArguments(
            output_dir=Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\results\tests', model_id+f'-{NFCS_REGION.name}'),
            save_strategy='epoch',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            learning_rate=metadata['optuna']['tunned_params']['learning_rate'],
            weight_decay=metadata['optuna']['tunned_params']['weight_decay'],
            num_train_epochs=metadata['optuna']['epochs'],
            per_device_train_batch_size=metadata['optuna']['tunned_params']['per_device_train_batch_size'],
            per_device_eval_batch_size=1,
            save_total_limit=1,
            remove_unused_columns=False,
            push_to_hub=False,
            fp16=False,  # float point 16 bit precision (instead of 32)
            load_best_model_at_end=True,
            metric_for_best_model=OBJECTIVE_METRIC,
            greater_is_better=GREATER_IS_BETTER,
            # save_only_model=True
        )

        if is_weighted:
            trainer = ClassWeightedTrainer(
                class_w=class_w,
                model=None,
                model_init=model_init,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=partial(compute_metrics, metrics=[load_metric(m, trust_remote_code=True) for m in METRICS]),
                train_dataset=ds['train'],
                eval_dataset=ds['test'],
                tokenizer=processor,
            )
        else:
            trainer = Trainer(
                # class_w=class_w,
                model=None,
                model_init=model_init,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=partial(compute_metrics, metrics=[load_metric(m, trust_remote_code=True) for m in METRICS]),
                train_dataset=ds['train'],
                eval_dataset=ds['test'],
                tokenizer=processor,
            )

        best_run = trainer.train()
        best_model = trainer.model

    # Eval loop
    # test_ds = InfantDataset(
    #     image_dir=Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\results\mosaic-2\ComArtefatoCompleto\masked'),
    #     csv_path=Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\src\Tatiany-Regioes\data\occlusion\Avaliadores_aparato-consenso-presente.csv'),
    #     nfcs_component=NFCS_REGION
    # )
    # if not MASKED and is_weighted and not is_frozen and DATASET.name == 'Full' and NFCS_REGION == NFCS.BA:
    #     pass

    # NOTA:
    # crops completos: E:\Pedro\Faculdade\FEI-Projeto_Dor\data\UNIFESP2\mosaic_and_manual_annotations\masked
    # imagens para recortar usando o labelme: E:\Pedro\Faculdade\FEI-Mestrado\Datasets\UNIFESP2\to_process
    # dataset de imagens completo: E:\Pedro\Faculdade\FEI-Mestrado\Datasets\UNIFESP2\completo

    test_ds_torch = InfantDataset(
        # ICOPE #
        # image_dir=Path(r'D:\ComputerScience\Mestrado\results\mosaic-2\iCOPE', 'masked' if MASKED else 'non_masked'),
        # csv_path=Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\src\Tatiany-Regioes\data\icope\icope_labels.csv'),
        # UNIFESP 2 #
        # image_dir=Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\data\UNIFESP2\mosaic_and_manual_annotations', 'masked' if MASKED else 'non_masked'),
        # csv_path=Path(r'E:\Pedro\Faculdade\FEI-Projeto_Dor\src\Tatiany-Regioes\data\occlusion\Avaliadores_aparato-consenso-presente-renamed.csv'),
        # FULL #
        image_dir=masked_images_path if MASKED else non_masked_images_path,
        csv_path=Path(full_dataset.csv_path),

        nfcs_component=NFCS_REGION,
    )
    test_ds = Dataset.from_generator(gen, gen_kwargs=({'ds': test_ds_torch}))
    test_ds.set_transform(partial(transforms, processor=processor, augmentations=Compose([])))
    # df_data = {region: np.full(ds['test'].num_rows, -1) for region in NFCS}
    # img_files_list = [test_data['files'] for test_data in ds['test']]

    # for test_data in tqdm(ds['test'], desc="Images evaluated"):
    for test_data in tqdm(test_ds, desc="Images evaluated"):
        X = torch.unsqueeze(test_data['pixel_values'], 0).cuda()
        y = test_data['labels']
        img_file = str(Path(test_data['files']).stem)

        with torch.no_grad():
            result = best_model(X)

        pred = np.argmax(result.logits.cpu().numpy())

        model_record = df_data[model_id]

        model_record['predictions'] = model_record.get('predictions', dict())
        model_record['labels'] = model_record.get('labels', dict())

        for key in model_record:
            model_img_record = model_record[key].get(img_file, np.full(len(NFCS), -1))

            model_record_region_idx = sorted(NFCS).index(NFCS_REGION)
            model_img_record[model_record_region_idx] = pred if key == 'predictions' else y
            model_record[key][img_file] = model_img_record

    print(f"model: '{model_id}' evaluated for region '{NFCS_REGION.name}'")


for model_config, model_config_data in df_data.items():
    save_path = Path(r"E:\Pedro\Faculdade\FEI-Projeto_Dor\src\Tatiany-Regioes\data\eval_results")
    save_path.mkdir(exist_ok=True, parents=True)

    with pd.ExcelWriter(save_path / f'{model_config}.xlsx') as writer:
        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        df = pd.DataFrame.from_dict(df_data[model_config]['predictions'], orient='index', columns=[r.name for r in sorted(NFCS)])
        df.sort_index().to_excel(writer, sheet_name="Predictions", index=True)
        df = pd.DataFrame.from_dict(df_data[model_config]['labels'], orient='index', columns=[r.name for r in sorted(NFCS)])
        df.sort_index().to_excel(writer, sheet_name="Labels", index=True)
