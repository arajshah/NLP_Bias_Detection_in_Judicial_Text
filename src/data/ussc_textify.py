from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class UsscTextifyConfig:

    include_cols: tuple[str, ...]
    sep: str = "; "
    kv_sep: str = ": "


DEFAULT_USSC_TEXTIFY = UsscTextifyConfig(
    include_cols=(
        "AGE",
        "NEWRACE",
        "MONSEX",
        "EDUCATN",
        "DISTRICT",
        "CIRCDIST",
        "CRIMHIST",
        "SENTYR",
        "CITIZEN",
        "CITWHERE",
        "NUMDEPEN",
        "CRIMLIV",
        "SENTMON",
        "ZONE",
    )
)


def row_to_text(row: pd.Series, cfg: UsscTextifyConfig = DEFAULT_USSC_TEXTIFY) -> str:
    parts: list[str] = []
    for c in cfg.include_cols:
        if c not in row.index:
            continue
        v = row[c]
        # Normalize NaNs/None
        if pd.isna(v):
            v = "Unknown"
        parts.append(f"{c}{cfg.kv_sep}{v}")
    return cfg.sep.join(parts)


def dataframe_to_text(df: pd.DataFrame, cfg: UsscTextifyConfig = DEFAULT_USSC_TEXTIFY) -> list[str]:
    return [row_to_text(r, cfg) for _, r in df.iterrows()]


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


