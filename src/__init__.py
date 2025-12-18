"""Bias Detection in Judicial Text - reusable ML/NLP components.

This package intentionally provides:
- Multi-task DistilBERT (classification + regression) for USSC-style structured data by
  serializing rows into text.
- Judicial/CAP text utilities: extraction, metadata joining, LDA topics, sentiment,
  and optional BERT-family classification/embeddings.
- Fairness metrics helpers (demographic parity, TPR/FPR by group).
"""


