"""
BERT Fine-Tuning Model for Aspect-Based Sentiment Analysis.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class ABSABert(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_name = pretrained_model_name
        self.num_classes = num_classes
        self.bert_classifier = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            classifier_dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        outputs = self.bert_classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return outputs.logits
