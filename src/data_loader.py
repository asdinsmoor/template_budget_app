from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_transactions(csv_paths: Iterable[str | Path]) -> pd.DataFrame:
    """
    Load one or more pre-cleaned CSV files and return them concatenated.

    Each CSV must already contain the expected columns
    (e.g. date, description, amount, currency, amount_aud, category, subcategory,
    exclude_flag, category_source, source_file, etc.). No additional normalisation
    or conversion is performed here; the caller is responsible for data hygiene.
    """
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    frames = []
    for raw_path in csv_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)
