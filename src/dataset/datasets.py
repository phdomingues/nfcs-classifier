"""
This module hosts all dataset classes.

The datasets in this module are subclasses of the
pytorch Dataset class.
"""

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import pandas as pd
from loguru import logger
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Generic dataset for image handling.

    This class extends the Pytorch Dataset class. It retrieves images with labels and file locations,
    verify labels csv sanity and remove unpaired labels (present in the csv file but image does not exist).
    """

    def __init__(
        self,
        image_dir: Path | str,
        csv_path: Path | str,
        transforms: Optional[Iterable[Callable[[PILImage], PILImage]]] = None,
    ) -> None:
        """
        Initialize the InfantDataset.

        Args:
            image_dir (Path | str): Path to the image directory.
            csv_path (Path | str): Path to the csv file containing file names and labels. \
                Filenames must be in the first column.
            transforms (optional, Iterable[Callable[[PILImage], PILImage]]): Any transforms to be applied to the \
                image before returning. Defaults to None.
        """
        super().__init__()
        self._csv_path = csv_path
        self.transforms = [] if transforms is None else transforms
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(
            self._csv_path,
            index_col=0,
        )
        self._remove_unpaired()

    def __len__(self) -> int:
        """Count the number of valid images are in the dataset.

        Returns:
            int: Number of valid images.
        """
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[PILImage, pd.Series, Path | str]:
        """Get image and labels for a given index in the dataset.

        Args:
            index (int): Index for the image.

        Returns:
            Tuple[Image, pd.Series]: Image (PIL), Labels.
        """
        labels = self.df.iloc[index]
        img_path = self._get_img_path(labels.name)
        img = Image.open(img_path)

        for transform in self.transforms:
            img = transform(img)

        return img, labels

    def add_transforms(self, *transforms: Callable[[PILImage], PILImage]) -> None:
        """Add transform operations to this dataset."""  # noqa
        self.transforms.extend(transforms)

    def _remove_unpaired(self) -> None:
        """Remove entries from the dataset that are not paired with an existing image file."""
        drop_rows = []
        for _, row in self.df.iterrows():
            try:
                self._get_img_path(row.name)
            except FileNotFoundError:
                drop_rows.append(row.name)

        if drop_rows:
            self.df.drop(
                drop_rows,
                axis=0,
                inplace=True,
            )
            logger.warning(
                f'Dropped images from dataset (file not found): {drop_rows} not found... dropping from dataset',
            )

    def _get_img_path(self, file_name: str, file_extension: str = '.*') -> str:
        """Find the image path based on file extension and file name.

        Args:
            file_name (str): Name for the image file (no file extension).
            file_extension (str): Extension for the image file.

        Raises:
            FileNotFoundError: Raised if the the file given by file_name was not found with any file extension.

        Returns:
            str: Path to the image.
        """
        try:
            return next(self.image_dir.glob(file_name+file_extension))
        except StopIteration:
            raise FileNotFoundError(self.image_dir / file_name)
