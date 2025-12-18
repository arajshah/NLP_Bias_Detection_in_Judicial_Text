#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.paths import get_paths


def extract_case_text_from_cap_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    opinion = soup.select_one("article.opinion")
    scope = opinion if opinion is not None else soup.select_one("section.casebody")
    if scope is None:
        return ""
    for foot in scope.select("aside.footnote"):
        foot.decompose()
    paras = [p.get_text(" ", strip=True) for p in scope.select("p")]
    return "\n".join([t for t in paras if t])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CAP-derived legal_text_data.csv from cap_data/*")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()
    cap = paths.cap_dir
    out_path = Path(args.out) if args.out else (paths.data_dir / "legal_text_data.csv")

    # Extract text from HTML
    rows = []
    for html_path in sorted((cap / "html").glob("*.html")):
        case_id = html_path.stem
        html = html_path.read_text(encoding="utf-8", errors="replace")
        text = extract_case_text_from_cap_html(html)
        soup = BeautifulSoup(html, "lxml")
        parties = soup.select_one("section.head-matter p.parties")
        title = parties.get_text(" ", strip=True) if parties else None
        rows.append({"case_id": case_id, "title": title, "opinion_text": text, "opinion_char_len": len(text)})
    df_text = pd.DataFrame(rows)

    # Metadata from JSON
    meta_rows = []
    for path in sorted((cap / "json").glob("*.json")):
        case_id = path.stem
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        citations = obj.get("citations") or []
        official_cite = None
        for c in citations:
            if isinstance(c, dict) and c.get("type") == "official":
                official_cite = c.get("cite")
                break
        court = obj.get("court") or {}
        juris = obj.get("jurisdiction") or {}
        meta_rows.append(
            {
                "case_id": case_id,
                "cap_case_numeric_id": obj.get("id"),
                "name": obj.get("name"),
                "name_abbreviation": obj.get("name_abbreviation"),
                "decision_date": obj.get("decision_date"),
                "docket_number": obj.get("docket_number"),
                "official_citation": official_cite,
                "court_name": court.get("name"),
                "court_abbrev": court.get("name_abbreviation"),
                "jurisdiction": juris.get("name_long") or juris.get("name"),
            }
        )
    df_meta = pd.DataFrame(meta_rows)

    df = df_text.merge(df_meta, on="case_id", how="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} | shape={df.shape}")


if __name__ == "__main__":
    main()


