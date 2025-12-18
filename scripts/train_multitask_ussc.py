#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.data.ussc_textify import DEFAULT_USSC_TEXTIFY, dataframe_to_text
from src.fairness import demographic_parity_by_group, tpr_fpr_by_group
from src.models.multitask_distilbert import DistilBertMultiTask
from src.paths import get_paths


def binary_disposit(disposit: str) -> int:
    if disposit in ["Jury trial", "Trial by judge or bench trial", "Guilty plea and trial (>1count)"]:
        return 1
    if disposit in ["Guilty plea", "Nolo contendere"]:
        return 0
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DistilBERT multi-task model on USSC-style structured dataset.")
    p.add_argument("--data", type=str, default=None, help="Path to cleaned_data_phase3_unencoded_DISPOSIT.csv")
    p.add_argument("--model", type=str, default="distilbert-base-uncased")
    p.add_argument("--out", type=str, default=None, help="Output directory for model artifacts")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protected_col", type=str, default="NEWRACE", help="Column to use for fairness slices")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = get_paths()
    data_path = Path(args.data) if args.data else (paths.data_dir / "cleaned_data_phase3_unencoded_DISPOSIT.csv")
    out_dir = Path(args.out) if args.out else (paths.models_dir / "multitask_distilbert_ussc")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    if "DISPOSIT" not in df.columns:
        raise ValueError("Expected DISPOSIT column for classification target.")
    if "SENTTOT_RAW" in df.columns:
        y_reg = pd.to_numeric(df["SENTTOT_RAW"], errors="coerce")
        reg_name = "SENTTOT_RAW"
    else:
        y_reg = pd.to_numeric(df["SENTTOT"], errors="coerce")
        reg_name = "SENTTOT"

    y_cls = df["DISPOSIT"].astype(str).apply(binary_disposit).astype(int)
    group = df[args.protected_col].astype(str) if args.protected_col in df.columns else pd.Series(["<missing>"] * len(df))

    texts = dataframe_to_text(df, DEFAULT_USSC_TEXTIFY)

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test, g_train, g_test = train_test_split(
        texts,
        y_cls.values,
        y_reg.values,
        group.values,
        test_size=0.2,
        random_state=args.seed,
        stratify=y_cls.values,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistilBertMultiTask(model_name=args.model)
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def batches(X, yc, yr, bs):
        for i in range(0, len(X), bs):
            yield X[i : i + bs], yc[i : i + bs], yr[i : i + bs]

    for epoch in range(args.epochs):
        running = 0.0
        for xb, ycb, yrb in batches(X_train, y_cls_train, y_reg_train, args.batch_size):
            enc = tok(
                list(xb),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            ycb_t = torch.tensor(ycb, device=device)
            yrb_t = torch.tensor(yrb, device=device)
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels_classification=ycb_t,
                labels_regression=yrb_t,
            )
            loss = out.loss
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
        print(f"epoch={epoch+1}/{args.epochs} train_loss={running:.4f}")

    # Evaluation
    model.eval()
    y_pred = []
    y_reg_pred = []
    with torch.no_grad():
        for xb, ycb, yrb in batches(X_test, y_cls_test, y_reg_test, args.batch_size):
            enc = tok(
                list(xb),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            prob = torch.softmax(out.logits_classification, dim=-1)
            pred = torch.argmax(prob, dim=-1).cpu().numpy()
            y_pred.append(pred)
            y_reg_pred.append(out.prediction_regression.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_reg_pred = np.concatenate(y_reg_pred, axis=0)

    acc = float(np.mean(y_pred == y_cls_test))
    mse = float(np.mean((y_reg_pred - y_reg_test) ** 2))
    print(f"test_acc={acc:.4f} | test_mse({reg_name})={mse:.4f}")

    dp = demographic_parity_by_group(y_pred=y_pred, group=g_test, positive_label=1)
    tprfpr = tpr_fpr_by_group(y_true=y_cls_test, y_pred=y_pred, group=g_test, positive_label=1)
    print("Demographic parity (pos rate) by group:")
    for k, v in sorted(dp.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v:.4f}")
    print("TPR/FPR by group:")
    for k, (tpr, fpr) in sorted(tprfpr.items(), key=lambda x: str(x[0])):
        print(f"  {k}: TPR={tpr} FPR={fpr}")

    # Save artifacts
    tok.save_pretrained(out_dir.as_posix())
    torch.save(model.state_dict(), (out_dir / "multitask_state_dict.pt").as_posix())
    (out_dir / "metrics.txt").write_text(
        f"test_acc={acc:.6f}\n"
        f"test_mse_{reg_name}={mse:.6f}\n",
        encoding="utf-8",
    )
    print(f"Saved model to: {out_dir}")


if __name__ == "__main__":
    main()


