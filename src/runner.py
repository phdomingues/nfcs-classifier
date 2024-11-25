"""Module for testing the developed code."""

from loguru import logger

from dataset.factory import add_vit_transforms, get_dataset_settings, get_dataset_from_config
from config import model_settings


@logger.catch()
def main():
    """Test the program."""
    print(f'Settings: {model_settings}')  # noqa
    ds_config = get_dataset_settings('UNIFESP')
    ds = get_dataset_from_config(ds_config)
    add_vit_transforms(ds)
    img, labels = ds[0]  # noqa
    img.show()


if __name__ == '__main__':
    main()
