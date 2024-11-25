"""This module contains all custom HuggingFace Trainers."""

from typing import Any, Optional, Tuple

import torch
from transformers import Trainer


class ClassWeightedTrainer(Trainer):
    """Custom trainer to add balanced loss functionality."""

    def __init__(self, class_w, *args, **kwargs):
        self.class_w = class_w
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: Any,
        inputs: Any,
        return_outputs: Optional[bool] = False,
    ) -> Tuple[Any, Any] | Any:
        """How the loss is computed by Trainer.

        Args:
            model (Any): Model.
            inputs (Any): Input data.
            return_outputs (bool, optional): True to return outputs with the loss. Defaults to False.

        Returns:
            Tuple[Any, Any] | Any: Loss, Outputs (If return_outputs is True)
        """
        labels = inputs.pop('labels')
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                self.class_w.tolist(),
                device=model.device,
            ),
        )
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss
