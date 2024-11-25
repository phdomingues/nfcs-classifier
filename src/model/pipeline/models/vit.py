

from typing import Any, Tuple

from config.model import model_settings
from transformers import ViTForImageClassification, ViTImageProcessor

def load_vit(model_name: str) -> Tuple[Any, Any]:
    model_path = next((model_settings.model_dir))
    # Processor (pre-process the images to fit the expected input)
    processor = ViTImageProcessor.from_pretrained(model_settings.model_dir)

    saved_model_path = TRAINED_MOODELS_PATH / f'{model_id}-{NFCS_REGION.name}'
    if saved_model_path.exists():
        best_model = ViTForImageClassification.from_pretrained(
            next((TRAINED_MOODELS_PATH / f'{model_id}-{NFCS_REGION.name}').glob('checkpoint*')),
            local_files_only=True,
        )
        best_model.cuda()
