import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .categorizer import categorize_transactions

EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"


@dataclass
class EvaluationSummary:
    file: str
    rows: int
    category_match_pct: float
    sub_category_match_pct: float
    exclude_flag_match_pct: float
    all_fields_match_pct: float


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", c.strip().lower()) for c in renamed.columns
    ]
    return renamed


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _string_equals(left: pd.Series, right: pd.Series) -> pd.Series:
    lnorm = left.fillna("").astype(str).str.strip().str.lower()
    rnorm = right.fillna("").astype(str).str.strip().str.lower()
    return lnorm == rnorm


def _boolify(series: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    def _convert(value) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return None
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        try:
            return bool(int(text))
        except ValueError:
            return None
    converted = series.apply(_convert)
    return converted


def _first_available(columns: List[str], options: List[str]) -> Optional[str]:
    columns_set = set(columns)
    for candidate in options:
        if candidate in columns_set:
            return candidate
    return None


def run_evaluation(batch_size: int = 10, preview_rows: int = 25) -> Dict[str, object]:
    eval_files = sorted(EVAL_DIR.glob("*.csv"))
    summaries: List[EvaluationSummary] = []
    previews: Dict[str, pd.DataFrame] = {}
    all_results: List[pd.DataFrame] = []

    for path in eval_files:
        df_raw = pd.read_csv(path)
        df_norm = _normalize_columns(df_raw)

        if "description" not in df_norm.columns:
            continue

        amount_col = "amount" if "amount" in df_norm.columns else None
        if amount_col is None and "amount_in_aud" in df_norm.columns:
            amount_col = "amount_in_aud"

        if amount_col:
            amounts_series = _to_numeric(df_norm[amount_col])
        else:
            amounts_series = pd.Series([0.0] * len(df_norm))

        descriptions = df_norm["description"].fillna("").tolist()
        amounts = amounts_series.tolist()

        cats, subs, exclude_flags, sources, confidences = categorize_transactions(
            descriptions,
            amounts,
            batch_size=batch_size,
        )

        result_df = pd.DataFrame(
            {
                "description": df_norm["description"].fillna(""),
                "amount": amounts_series,
                "predicted_category": cats,
                "predicted_sub_category": subs,
                "predicted_exclude_flag": [bool(x) for x in exclude_flags],
                "prediction_source": sources,
                "predicted_confidence": confidences,
            }
        )

        true_category_col = _first_available(
            df_norm.columns.tolist(),
            ["true_category", "category"],
        )
        true_sub_category_col = _first_available(
            df_norm.columns.tolist(),
            ["true_sub_category", "true_subcategory", "sub_category", "subcategory"],
        )
        true_exclude_col = _first_available(
            df_norm.columns.tolist(),
            ["true_exclude_flag", "exclude_flag"],
        )

        if true_category_col is None or true_sub_category_col is None or true_exclude_col is None:
            continue

        result_df["true_category"] = df_norm[true_category_col]
        result_df["true_sub_category"] = df_norm[true_sub_category_col]
        result_df["true_exclude_flag"] = _boolify(df_norm[true_exclude_col])

        result_df["category_match"] = _string_equals(
            result_df["predicted_category"],
            result_df["true_category"],
        )
        result_df["sub_category_match"] = _string_equals(
            result_df["predicted_sub_category"],
            result_df["true_sub_category"],
        )

        truth_exclude_filled = result_df["true_exclude_flag"].fillna(False)
        result_df["exclude_flag_match"] = (
            result_df["predicted_exclude_flag"] == truth_exclude_filled
        )

        result_df["all_fields_match"] = (
            result_df["category_match"]
            & result_df["sub_category_match"]
            & result_df["exclude_flag_match"]
        )

        rows = len(result_df)
        if rows == 0:
            continue

        summary = EvaluationSummary(
            file=path.name,
            rows=rows,
            category_match_pct=round(result_df["category_match"].mean() * 100, 2),
            sub_category_match_pct=round(result_df["sub_category_match"].mean() * 100, 2),
            exclude_flag_match_pct=round(result_df["exclude_flag_match"].mean() * 100, 2),
            all_fields_match_pct=round(result_df["all_fields_match"].mean() * 100, 2),
        )

        summaries.append(summary)

        columns = [
            "description",
            "amount",
            "true_category",
            "predicted_category",
            "category_match",
            "true_sub_category",
            "predicted_sub_category",
            "sub_category_match",
            "true_exclude_flag",
            "predicted_exclude_flag",
            "exclude_flag_match",
            "all_fields_match",
            "prediction_source",
            "predicted_confidence",
        ]

        file_results = result_df[columns].copy()
        file_results.insert(0, "file", path.name)
        file_results_sorted = file_results.sort_values(
            ["all_fields_match", "category_match", "sub_category_match"],
            ascending=[True, True, True],
        )
        previews[path.name] = file_results_sorted.head(preview_rows)
        all_results.append(file_results_sorted)

    return {
        "summaries": [asdict(summary) for summary in summaries],
        "previews": previews,
        "all_results": pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(),
    }
