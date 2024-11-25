"""Module that contains all custom exceptions related to datasets."""


class InvalidDatasetError(ValueError):
    """Exception for loading datasets that are invalid."""

    def __init__(self, dataset_name: str, *args: object) -> None:
        """Initialize the InvalidDataset object.

        Args:
            dataset_name (str): Name given for the invalid dataset.
            *args: Arguments passed to ValueError.
        """
        super().__init__(
            f'Dataset named "{dataset_name}" not listed as a valid dataset.',
            *args,
        )
