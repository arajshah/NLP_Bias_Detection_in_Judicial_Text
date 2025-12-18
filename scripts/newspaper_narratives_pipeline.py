#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.paths import get_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a lightweight NLP pipeline on a user-supplied newspaper narrative CSV."
        )
    )
    p.add_argument("--csv", required=True, help="Path to CSV with narrative text")
    p.add_argument("--text_col", default="text", help="Name of the text column")
    p.add_argument("--out_dir", default=None, help="Directory to write artifacts (topics, doc-topic matrix)")
    p.add_argument("--num_topics", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()
    out_dir = Path(args.out_dir) if args.out_dir else (paths.reports_dir / "newspaper_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")
    texts = df[args.text_col].astype(str).fillna("").tolist()

    # LDA topics (bag-of-words)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=args.num_topics, random_state=42)
    doc_topics = lda.fit_transform(dtm)

    # Save top terms per topic
    vocab = vectorizer.get_feature_names_out()
    topic_terms = []
    for topic_idx, topic in enumerate(lda.components_):
        top_ids = topic.argsort()[-15:][::-1]
        terms = [vocab[i] for i in top_ids]
        topic_terms.append({"topic": topic_idx, "top_terms": ", ".join(terms)})

    pd.DataFrame(topic_terms).to_csv(out_dir / "lda_topics.csv", index=False)
    pd.DataFrame(doc_topics, columns=[f"topic_{i}" for i in range(args.num_topics)]).to_csv(
        out_dir / "doc_topic_matrix.csv", index=False
    )

    print(f"Wrote topics: {(out_dir / 'lda_topics.csv')}")
    print(f"Wrote doc-topic matrix: {(out_dir / 'doc_topic_matrix.csv')}")


if __name__ == "__main__":
    main()


