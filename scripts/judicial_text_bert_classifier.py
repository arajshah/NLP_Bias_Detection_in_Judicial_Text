#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.paths import get_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a BERT-family model on CAP-derived judicial text CSV.")
    p.add_argument("--csv", type=str, default=None, help="Path to data/legal_text_data.csv")
    p.add_argument("--text_col", type=str, default="opinion_text")
    p.add_argument("--label_col", type=str, default="BIAS_LABEL", help="Binary label column (proxy or gold)")
    p.add_argument("--model", type=str, default="distilbert-base-uncased")
    p.add_argument("--out", type=str, default=None, help="Output directory")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


def main() -> None:
    args = parse_args()
    paths = get_paths()
    csv_path = Path(args.csv) if args.csv else (paths.data_dir / "legal_text_data.csv")
    out_dir = Path(args.out) if args.out else (paths.models_dir / "judicial_text_distilbert")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    texts = df[args.text_col].astype(str).fillna("").tolist()
    labels = df[args.label_col].astype(int).values

    train_text, val_text, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    train_enc = tok(train_text, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    val_enc = tok(val_text, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")

    train_ds = SimpleTextDataset(train_enc, y_train)
    val_ds = SimpleTextDataset(val_enc, y_val)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer_out"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        pred = np.argmax(logits, axis=-1)
        acc = float(np.mean(pred == y_true))
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved model: {out_dir}")


if __name__ == "__main__":
    main()


