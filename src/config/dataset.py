"""
This module sets up all dataset configuration.

It reads all configs from enviromnent variables and a
.env file using Pydantic.
"""

from pathlib import Path
from typing import List

from pydantic import Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings

from utils.nfcs import NFCS


class DatasetSettings(BaseSettings):
    """Dataset configuration settings for the application.

    Args:
        csv_path (Path): Path to the csv file containing labels.
        image_dir (Path): Path to the directory containing the images.
    """

    csv_path: Path
    image_dir: Path
    label_cols: List[str] | None = Field(default=None)
    # skip_label: Any = Field(default=None)  # noqa
    _name: str = PrivateAttr('')

    @field_validator('csv_path')
    @classmethod
    def csv_path_validator(cls, csv_path: Path) -> Path:
        """Check if the csv_path is valid and if the file exists.

        Args:
            csv_path (Path): Path to the csv file.

        Raises:
            ValueError: Raised if the file does not exist or if is not a csv.

        Returns:
            Path: Valid path.
        """
        if csv_path.suffix != '.csv' or not csv_path.is_file():
            raise ValueError(f'Invalid csv path: "{csv_path}"')
        return csv_path

    @field_validator('image_dir')
    @classmethod
    def image_dir_validator(cls, image_dir: Path) -> Path:
        """Check if the image_dir path is valid and if the directory exists.

        Args:
            image_dir (Path): Path to the directory.

        Raises:
            ValueError: Raised if the directory does not exist.

        Returns:
            Path: Valid path.
        """
        if not image_dir.is_dir():
            raise ValueError(f'Invalid image directory path: "{image_dir}"')
        return image_dir

    @property
    def name(self) -> str:  # noqa
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name  # noqa

    class Config:
        """BaseSettings configurations."""

        env_file_encoding = 'utf-8'
        env_nested_delimiter = '__'
        extra = 'forbid'
        protected_namespaces = ('settings_',)


class NFCSDatasetSettings(DatasetSettings):
    """NFCS Dataset configuration settings."""

    nfcs_regions: List[str]

    @field_validator('nfcs_regions')
    @classmethod
    def nfcs_regions_validator(cls, nfcs_regions: List[str]) -> List[NFCS]:
        """Check if the nfcs_regions provided are valid and convert them to NFCS enum.

        Args:
            nfcs_regions (Path): Path to the directory.

        Returns:
            List[NFCS]: NFCS enum.
        """
        return [NFCS[region] for region in nfcs_regions]
