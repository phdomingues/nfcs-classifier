"""This module contains a factory design pattern to create and manage dataset objects."""


from functools import partial
from typing import List

from torch.utils.data import Dataset

from config.dataset import DatasetSettings
from dataset.datasets import ImageDataset
from dataset.exceptions import InvalidDatasetError
from model.pipeline.transforms import resize_pad


def available_datasets() -> List[str]:
    """Return a list of all available datasets.

    Returns:
        List[str]: Available dataset options.
    """
    return [
        'UNIFESP',
        'UNIFESP-NFCS',
        'FULL',
        'FULL-NFCS',
    ]


def get_dataset_settings(dataset: str) -> DatasetSettings:
    """Load the settings for a specified dataset.

    Args:
        dataset (str): Dataset name.

    Raises:
        InvalidDatasetError: Raised if the dataset is not listed as valid.

    Returns:
        DatasetSettings: Settings loaded from the .env file.
    """
    if dataset not in available_datasets():
        raise InvalidDatasetError(dataset)
    ds = DatasetSettings(_env_file=f'config/dataset_config/.{dataset}.env')
    ds.name = dataset
    return ds


def get_dataset_from_config(dataset_config: DatasetSettings) -> Dataset:
    """Instantiate a dataset based on the given settings.

    Args:
        dataset_config (DatasetSettings): Settings for the dataset to be created.

    Raises:
        InvalidDatasetError: Raised if the dataset is not listed as valid.

    Returns:
        Dataset: Instance of the dataset.
    """
    if dataset_config.name not in available_datasets():
        raise InvalidDatasetError(dataset_config.name)
    return ImageDataset(
        image_dir=dataset_config.image_dir,
        csv_path=dataset_config.csv_path,
    )


def get_dataset(dataset_name: str) -> Dataset:
    """Get a dataset from name.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        Dataset: Corresponding Dataset.
    """
    ds_config = get_dataset_settings(dataset_name)
    return get_dataset_from_config(ds_config)


def add_vit_transforms(dataset: ImageDataset) -> None:
    """Add all transformations necessary for the ViT model.

    Args:
        dataset (ImageDataset): Dataset to apply the transforms.
    """
    vit_required_size = 224

    dataset.add_transforms(
        partial(resize_pad, smaller_side_size=vit_required_size),
    )
