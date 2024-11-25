import datetime
import json
import optuna
import pickle

from datasets import load_metric, list_metrics, Dataset
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
    name='Full', csv_path=r"D:\ComputerScience\Mestrado\data\UNIFESP\NEW_GT.csv")
unifesp_dataset = DatasetMeta(
    name='UNIFESP', csv_path=r"D:\ComputerScience\Mestrado\data\UNIFESP\NEW_GT_UNIFESP.csv"
)

for dataset_meta in tqdm([unifesp_dataset, full_dataset], desc='Dataset'):
    for apply_mask in tqdm([True, False], desc='Apply mask'):
        for run_nfcs_region in tqdm(NFCS, desc='NFCS region'):
            # Datetime
            now = datetime.datetime.now()
            now_str = now.strftime('%d%m%y-%H%M')
            # General
            MOCK = True # Marks this as a mock run. Mock runs are saved in a test directory to avoid poluting the results directory
            DESCRIPTION = 'Non weighted loss ; All layers trained' # Use this string to describe any important points of the experiment. This will be saved along with the results
            # Model
            DATASET:DatasetMeta   = dataset_meta
            NFCS_REGION:NFCS      = run_nfcs_region
            MASKED:bool           = apply_mask # If true use the masked crops if false use the cropped only
            VALIDATION_SIZE:float = 0.15 # % of images used for validation
            METRICS:List[str]     = ["accuracy", "f1"] # For a list of available metrics call the function list_metrics()
            OBJECTIVE_METRIC      = "eval_f1" # Metric that will be optimized by optuna
            GREATER_IS_BETTER     = True # Relative to the objective metric
            DIRECTION             = "maximize" if GREATER_IS_BETTER else "minimize"
            # Hyperparam tunning
            NUM_TRIALS:int   = 30 # 20
            TRAIN_EPOCHS:int = 20 # 10
            # Model path to download from huggingface
            model_name_or_path:str = 'google/vit-base-patch16-224-in21k'


            # Dataset paths
            output_dir = r'E:\Pedro\Faculdade\FEI-Projeto_Dor\results\Optuna' + f"{'/mock' if MOCK else ''}/ViT/{DATASET.name}/{'non_' if not MASKED else ''}masked/{NFCS_REGION.name}/{now_str}"
            csv_path = DATASET.csv_path
            masked_images_path = r"D:\ComputerScience\Mestrado\data\masked_crops"
            non_masked_images_path = r"D:\ComputerScience\Mestrado\data\non_masked_crops"
            # Create output dirs if needed
            Path(output_dir).mkdir(parents=True, exist_ok=True)


            # Image augmentations
            augmentations = Compose([ # https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html
                RandomAffine(degrees=(-45,45), translate=(0.1, 0.3), scale=(0.7, 0.9)),
                ColorJitter(brightness=.4, hue=0.3, saturation=0.6), # not contrast, or the background will be afected too
                GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
                RandomHorizontalFlip(p=0.15),
                RandomVerticalFlip(p=0.15),
            ])
            # Optuna funcs
            def optuna_hp_space(trial):
                return {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
                    "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
                    "weight_decay": trial.suggest_categorical("weight_decay", [0, 0.1])
                }
            def compute_objective(metrics):
                return metrics[OBJECTIVE_METRIC]

            # Processor (pre-process the images to fit the expected input)
            processor = ViTImageProcessor.from_pretrained(model_name_or_path)
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
                # Freeze parameters for the base model (don't freeze the classification head)
                # for param in model.base_model.parameters():
                #     param.requires_grad = False
                #####################################
                
                return model

            # Dataset
            torch_ds = InfantDataset(
                image_dir=masked_images_path if MASKED else non_masked_images_path,
                csv_path=csv_path,
                nfcs_component=NFCS_REGION
            )
            # Cast dataset from pytorch to huggingface
            ds = Dataset.from_generator(gen, gen_kwargs=({'ds':torch_ds}))
            # Set train test split
            # The train and test are beeing reversed and shuffle is set to False so the UNIFESP part of the full dataset is the only one used for validation 
            # (since it is loaded first and the train_test_split sets the first images as training by default)
            ds = ds.train_test_split(test_size=1-VALIDATION_SIZE, shuffle=False)
            ds['train'], ds['test'] = ds['test'], ds['train']
            # Add augmentations and pre-process
            ds['train'].set_transform(partial(transforms, processor=processor, augmentations=augmentations))
            ds['test'].set_transform(partial(transforms, processor=processor, augmentations=Compose([]))) # No augmentations in the test set

            total = ds['train'].num_rows
            class_w = np.zeros(2)
            for data in ds['train']:
                class_w[data['labels']] += 1
            class_w /= total

            #### ====== Save images used for training and validation ====== ####
            imgs_record = {'file': [], 'label': [], 'set': []}
            for set_name, set_obj in ds.items():
                for data in set_obj:
                    imgs_record['file'].append(data['files'])
                    imgs_record['label'].append(data['labels'])
                    imgs_record['set'].append(set_name)
            df = pd.DataFrame(imgs_record)
            df.to_csv(f'{output_dir}/dataset_split.csv')

            # HuggingFace trainer
            training_args = TrainingArguments(
                output_dir=output_dir,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                logging_strategy='epoch',
                # learning_rate=config.learning_rate,
                # weight_decay=config.weight_decay,
                num_train_epochs=TRAIN_EPOCHS,
                # per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=1,
                save_total_limit=1,
                remove_unused_columns=False,
                push_to_hub=False,
                fp16=False, # float point 16 bit precision (instead of 32)
                load_best_model_at_end=True,
                metric_for_best_model=OBJECTIVE_METRIC,
                greater_is_better=GREATER_IS_BETTER,
                save_only_model=True
            )

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

            best_run = trainer.hyperparameter_search(
                direction="maximize",
                backend="optuna",
                hp_space=optuna_hp_space,
                n_trials=NUM_TRIALS,
                compute_objective=compute_objective,
            )

            # Save results
            study = trainer._trial.study
            # Save study
            optuna.copy_study(
                from_study_name=study.study_name,
                from_storage=study._storage,
                to_storage=f"sqlite:///{output_dir}/study.db",
                to_study_name="study"
            )
            # Save sampler state
            with open(f"{output_dir}/sampler.pkl", "wb") as fout:
                pickle.dump(study.sampler, fout)
            # Save pruner state
            with open(f"{output_dir}/pruner.pkl", "wb") as fout:
                pickle.dump(study.pruner, fout)
            # Run Summary
            set_split_summary = {
                'train': {
                    'image_count': {
                        '0': 0,
                        '1': 0
                    }
                },
                'test': {
                    'image_count': {
                        '0': 0,
                        '1': 0
                    }
                }
            }
            for ds_set in set_split_summary:
                for data in ds[ds_set]:
                    set_split_summary[ds_set]['image_count'][str(data['labels'])] += 1
                total = sum(set_split_summary[ds_set]['image_count'].values())
                ratio0 = round(set_split_summary[ds_set]['image_count']['0'] / total * 100)
                set_split_summary[ds_set]['ratio'] = f'{ratio0}/{100-ratio0}'
            summary = {
                'optuna': {
                    'best_run_id': best_run.run_id,
                    'objective': OBJECTIVE_METRIC,
                    'direction': DIRECTION,
                    'optimized_objective': best_run.objective,
                    'trials': NUM_TRIALS,
                    'epochs': TRAIN_EPOCHS,
                    'tunned_params': best_run.hyperparameters,
                },
                'dataset': DATASET.name,
                'nfcs': NFCS_REGION.name,
                'masked': MASKED,
                'val_split': VALIDATION_SIZE,
                'train_val_details': set_split_summary,
                'model': model_name_or_path,
                'pickle_version': pickle.format_version,
                'timestamp': now.isoformat(timespec='minutes'),
                'description': DESCRIPTION
            }
            with open(f'{output_dir}/summary.json', "w") as f:
                json.dump(summary, f, indent=2)