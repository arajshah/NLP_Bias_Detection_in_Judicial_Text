from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass(frozen=True)
class MultiTaskOutputs:
    logits_classification: torch.Tensor  # (B, 2)
    prediction_regression: torch.Tensor  # (B,)
    loss: torch.Tensor | None = None
    loss_cls: torch.Tensor | None = None
    loss_reg: torch.Tensor | None = None


class DistilBertMultiTask(nn.Module):
    """Multi-task model: binary classification + regression on a shared DistilBERT encoder.

    Designed to support the CV claim: a transformer-based multi-task pipeline
    for trial occurrence (classification) and sentencing length (regression).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.2,
        regression_loss: str = "mse",
        loss_weight_cls: float = 1.0,
        loss_weight_reg: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.loss_weight_cls = loss_weight_cls
        self.loss_weight_reg = loss_weight_reg

        cfg = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=cfg)
        hidden = cfg.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden, num_labels)
        self.reg_head = nn.Linear(hidden, 1)

        self._cls_loss_fn = nn.CrossEntropyLoss()
        if regression_loss == "mae":
            self._reg_loss_fn = nn.L1Loss()
        else:
            self._reg_loss_fn = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_classification: torch.Tensor | None = None,
        labels_regression: torch.Tensor | None = None,
    ) -> MultiTaskOutputs:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBERT: take [CLS] token representation at position 0
        pooled = enc.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        logits_cls = self.cls_head(pooled)  # (B,2)
        pred_reg = self.reg_head(pooled).squeeze(-1)  # (B,)

        loss = None
        loss_cls = None
        loss_reg = None
        if labels_classification is not None:
            loss_cls = self._cls_loss_fn(logits_cls, labels_classification.long())
        if labels_regression is not None:
            loss_reg = self._reg_loss_fn(pred_reg.float(), labels_regression.float())

        if loss_cls is not None or loss_reg is not None:
            loss = torch.tensor(0.0, device=logits_cls.device)
            if loss_cls is not None:
                loss = loss + self.loss_weight_cls * loss_cls
            if loss_reg is not None:
                loss = loss + self.loss_weight_reg * loss_reg

        return MultiTaskOutputs(
            logits_classification=logits_cls,
            prediction_regression=pred_reg,
            loss=loss,
            loss_cls=loss_cls,
            loss_reg=loss_reg,
        )


