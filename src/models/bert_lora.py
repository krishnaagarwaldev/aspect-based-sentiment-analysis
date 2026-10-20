"""
BERT with LoRA (Low-Rank Adaptation) via Hugging Face PEFT for ABSA.
"""

import logging
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class ABSABertLoRA(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list = None,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.num_classes = num_classes

        if target_modules is None:
            target_modules = ["query", "value"]

        base_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.model = get_peft_model(base_model, peft_config)
        self.log_trainable_parameters()

    def log_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        pct = 100 * trainable_params / all_params
        logger.info(
            f"BERT-LoRA Trainable params: {trainable_params:,} / {all_params:,} ({pct:.2f}%)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return outputs.logits
