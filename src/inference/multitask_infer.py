from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.models.multitask_distilbert import DistilBertMultiTask


@dataclass(frozen=True)
class MultiTaskPredictions:
    y_cls_pred: np.ndarray  # (N,)
    y_cls_prob: np.ndarray  # (N,2)
    y_reg_pred: np.ndarray  # (N,)


def load_multitask_model(model_dir: Path, device: str | None = None) -> tuple[DistilBertMultiTask, AutoTokenizer, torch.device]:
    """Load a saved multitask model (state_dict + tokenizer)."""
    model_dir = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir.as_posix())
    model = DistilBertMultiTask(model_name=model_dir.as_posix())

    state_path = model_dir / "multitask_state_dict.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu"))

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()
    return model, tok, dev


@torch.no_grad()
def predict_texts(
    model: DistilBertMultiTask,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 256,
) -> MultiTaskPredictions:
    y_prob: list[np.ndarray] = []
    y_cls: list[np.ndarray] = []
    y_reg: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        prob = torch.softmax(out.logits_classification, dim=-1).cpu().numpy()
        pred_cls = np.argmax(prob, axis=-1)
        pred_reg = out.prediction_regression.cpu().numpy()
        y_prob.append(prob)
        y_cls.append(pred_cls)
        y_reg.append(pred_reg)

    return MultiTaskPredictions(
        y_cls_pred=np.concatenate(y_cls, axis=0),
        y_cls_prob=np.concatenate(y_prob, axis=0),
        y_reg_pred=np.concatenate(y_reg, axis=0),
    )


