"""
This module sets up all model configuration.

It reads all configs from enviromnent variables and a
.env file using Pydantic.
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

from dataset.exceptions import InvalidDatasetError
# from dataset.factory import available_datasets
from utils.nfcs import NFCS


class OptimizationSettings(BaseModel):
    """Configuration settings related to hyperparameter optimization.

    Args:
        direction (str): Either "maximize" or "minimize".
        trials (int): Number of optimization trials.
    """

    direction: str
    trials: int


class TrainingSettings(BaseModel):
    """Settings related to training.

    Args:
        dataset (str): Name of the dataset to be used for training.
        greater_is_better (bool): True if the objective_metric should be maximized.
        num_epochs (int): Number of epochs to train.
        objective_metric (str): Metric used to compare models.
        optimization (OptimizationSettings): Metrics related to the optimization process.
        val_split (float): Split of the dataset (in %) used for validation.
    """

    dataset: str
    greater_is_better: bool
    num_epochs: int
    objective_metric: str
    optimization: OptimizationSettings
    val_split: float

    # @field_validator('dataset')
    # @classmethod
    # def name_must_contain_space(cls, dataset: str) -> str:
    #     """Validate the DATASET field in .env.

    #     Args:
    #         dataset (str): Dataset name.

    #     Raises:
    #         InvalidDatasetError: Raised if the dataset is not listed as valid by the factory.

    #     Returns:
    #         str: The dataset.
    #     """
    #     if dataset not in available_datasets():
    #         raise InvalidDatasetError(dataset)
    #     return dataset


class ModelSettings(BaseSettings):
    """Model configuration settings for the application.

    Args:
        target_regions (NFCS): NFCS region to process.
        mask (bool): True if the NFCS regions should be masked on the image.
        base_model (str): Path to the base model (HuggingFace).
        training (TrainingSettings): Settings related to training.
        cuda (bool): Load model with cuda.
        model_dir (Path): Path to where all models are saved.
    """

    target_regions: List[NFCS]
    mask: bool
    base_model: str
    training: TrainingSettings
    cuda: bool
    model_dir: Path

    class Config:
        """BaseSettings configurations."""

        env_file = 'config/.env'
        env_file_encoding = 'utf-8'
        env_nested_delimiter = '__'
        extra = 'ignore'
        protected_namespaces = ('settings_',)

    @field_validator('target_regions', mode='before')
    @classmethod
    def validate_enum_name(cls, target_regions: list[str]) -> NFCS:
        """
        Validate the target_region field to enable the enumerator to be set by name and not by value.

        Args:
            target_regions (list[str]): Value read from the .env file as the target_region.

        Raises:
            ValueError: Raised if nfcs_region is not a valid option.

        Returns:
            NFCS: The NFCS (enum) region.
        """
        for region_idx, region in enumerate(target_regions):
            try:
                target_regions[region_idx] = NFCS[region]
            except KeyError:
                nfcs_names = [region.name for region in NFCS]
                raise ValueError(f'Invalid region name: {region}, expected one of {nfcs_names}')

        return target_regions


model_settings = ModelSettings()
