from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reversal_mm_constants import EVENT_DTYPE


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ensure_event_dtype(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == EVENT_DTYPE:
        return arr
    if arr.dtype.names is None:
        raise ValueError("Expected a structured event array.")
    out = np.empty(arr.shape[0], dtype=EVENT_DTYPE)
    for col in EVENT_DTYPE.names:
        out[col] = arr[col]
    return out

def load_hbt_npz(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as zf:
        arr = zf["data"] if "data" in zf.files else zf[zf.files[0]]
    return _ensure_event_dtype(arr)


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<empty>"
    return df.to_string(index=False, float_format=lambda x: f"{x:.6f}")
