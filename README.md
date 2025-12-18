# Bias Detection in Judicial Text

This repo is a lightweight ML/NLP exploration for studying **bias/fairness** in:

- **Structured sentencing outcomes** (tabular ML): `DISPOSIT` (disposition) and `SENTTOT` (sentence length)
- **Judicial text/opinion data** (NLP): CAP (Caselaw Access Project) HTML/JSON + metadata

The project includes both notebook exploration and reusable pipeline components under `src/` and `scripts/`.

## Directory layout

- `src/`
  - reusable components (paths, fairness metrics, USSC row→text serialization, multi-task DistilBERT)
- `scripts/`
  - runnable CLI scripts to train/evaluate models and build datasets
- `cap_data/`
  - `html/`: CAP opinion HTML (extractable opinion text)
  - `json/`: CAP opinion JSON (case metadata + citations + structure)
  - `metadata/`: CAP metadata (`CasesMetadata.json` is a list; `VolumeMetadata.json` is a dict)
- `notebooks/`: analysis pipeline

## Notebook run order

1. `notebooks/2_Data_Exploration_Final.ipynb`
2. `notebooks/3_Feature_Engineering.ipynb`
3. `notebooks/4_Baseline_Modeling.ipynb`
   - Baseline classification/regression + simple group metrics
4. `notebooks/6_NLP_CAP.ipynb`
   - Extracts CAP metadata + opinion text, basic NLP exploration
5. `notebooks/5_Advanced_Modeling_and_Fairness_Techniques.ipynb`
   - Guarded scaffold (only runs sections if required inputs exist)

## Pipelines

- **Multi-task DistilBERT on sentencing data**: joint **trial occurrence classification** + **sentence length regression**, with group fairness slices (demographic parity, TPR/FPR). See `scripts/train_multitask_ussc.py` and `src/models/multitask_distilbert.py`.
- **Judicial text NLP on CAP opinions**: dataset build from CAP artifacts, plus sentiment + LDA topic modeling and an optional DistilBERT classifier. See `scripts/build_legal_text_dataset.py`, `notebooks/6_NLP_CAP.ipynb`, and `scripts/judicial_text_bert_classifier.py`.

## Environment

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- NLP components (`transformers`, `torch`, `spacy`, `xgboost`) are optional but included.
- If running BERT sections, you may need model downloads/caches.


