# --- src/app.py (Streamlit UI only; categorizer kept separate) ---
import logging
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from src.categorizer import categorize_transactions
from src.evaluator import run_evaluation
from src.summarizer import summarize_spend, plot_spend

log_level = os.getenv("APP_LOG_LEVEL", "INFO").upper()
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
logger = logging.getLogger(__name__)

batch_size_env = os.getenv("APP_LLM_BATCH_SIZE")
default_batch_size = 100
if batch_size_env:
    try:
        default_batch_size = max(1, int(batch_size_env))
    except ValueError:
        logger.warning(
            "Invalid APP_LLM_BATCH_SIZE '%s'; defaulting to %d.",
            batch_size_env,
            default_batch_size,
        )

st.set_page_config(page_title="Personal Budget Assistant", layout="centered")
st.title("💰 Personal Budget Assistant")

with st.sidebar:
    st.header("Settings")
    batch_size = st.number_input(
        "LLM batch size",
        min_value=1,
        max_value=200,
        value=default_batch_size,
        step=1,
        format="%d",
        help="Number of uncategorized transactions sent to the LLM in each request.",
    )
batch_size = int(batch_size)
os.environ["APP_LLM_BATCH_SIZE"] = str(batch_size)
logger.info("Using LLM batch size %d.", batch_size)

uploaded_files = st.file_uploader(
    "Upload one or more CSV files with your transactions",
    accept_multiple_files=True,
    type=["csv"]
)

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f)
        df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c.strip().lower()) for c in df.columns]
        dfs.append(df)
        logger.info("Loaded file '%s' with %d rows.", f.name, len(df))
        with st.expander(f"Preview: {f.name} ({len(df)} rows) - showing first 5"):
            st.dataframe(df.head(5))
    df = pd.concat(dfs, ignore_index=True)

    st.write(f"Loaded {len(df)} transactions.")
    logger.info("Combined dataframe has %d rows.", len(df))
    # (Moved categorized preview below after categories are ensured)

    # --- Categorization ---
    # Ensure required columns exist
    if "category" not in df.columns:
        df["category"] = pd.NA
    if "subcategory" not in df.columns:
        df["subcategory"] = pd.NA
    if "exclude_flag" not in df.columns:
        df["exclude_flag"] = False
    if "category_source" not in df.columns:
        df["category_source"] = "default"
    if "llm_confidence" not in df.columns:
        df["llm_confidence"] = pd.NA

    df["exclude_flag"] = (
        df["exclude_flag"]
        .fillna(False)
        .map(lambda x: str(x).strip().lower() in {"true", "1", "yes"} if pd.notna(x) else False)
        .astype(bool)
    )

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    existing_mask = df["category"].notna() & (df["category"].astype(str).str.strip() != "")
    df.loc[existing_mask & (df["category_source"].astype(str).str.strip() == ""), "category_source"] = "provided"
    df.loc[existing_mask & df["category_source"].isna(), "category_source"] = "provided"
    df.loc[existing_mask & (df["subcategory"].astype(str).str.strip() == ""), "subcategory"] = "General"
    df.loc[existing_mask & df["subcategory"].isna(), "subcategory"] = "General"

    missing_mask = ~existing_mask
    if missing_mask.any():
        logger.info("Categorizing %d transactions via rules/LLM.", missing_mask.sum())
        descriptions = df.loc[missing_mask, "description"].fillna("").tolist()
        if "amount" in df.columns:
            amount_series = df.loc[missing_mask, "amount"]
            if pd.api.types.is_numeric_dtype(amount_series):
                amounts = amount_series.fillna(0).tolist()
            else:
                amounts = pd.to_numeric(amount_series, errors="coerce").fillna(0).tolist()
        else:
            amounts = [0.0] * missing_mask.sum()

        cats, subs, exclude_flags, sources, confidences = categorize_transactions(
            descriptions,
            amounts,
            batch_size=batch_size,
        )
        df.loc[missing_mask, "category"] = cats
        df.loc[missing_mask, "subcategory"] = subs
        df.loc[missing_mask, "exclude_flag"] = exclude_flags
        df.loc[missing_mask, "category_source"] = sources
        df.loc[missing_mask, "llm_confidence"] = pd.Series(confidences, index=df.index[missing_mask])
    else:
        logger.info("All transactions already have categories; skipping categorization.")

    df["subcategory"] = df["subcategory"].fillna("General")
    df["category_source"] = df["category_source"].fillna("default")
    df["llm_confidence"] = pd.to_numeric(df["llm_confidence"], errors="coerce")

    with st.expander("Preview: Categorization - key columns (first 10)"):
        preview_cols = [
            c
            for c in [
                "date",
                "description",
                "amount_in_aud",
                "category",
                "subcategory",
                "category_source",
                "exclude_flag",
                "llm_confidence",
            ]
            if c in df.columns
        ]
        st.dataframe(df[preview_cols].head(10) if preview_cols else df.head(10))

    summary = summarize_spend(df, by="category")
    if summary.empty:
        logger.warning("Spend summary is empty; skipping plot and report.")
        st.warning("No categorized transactions to summarize yet.")
    else:
        plot_spend(summary)

        st.subheader("Spending Summary")
        st.bar_chart(summary)

        # st.subheader("Markdown Report")
        # report_md = make_markdown_report(summary)
        # st.markdown(report_md)

        # st.download_button(
        #     label="Download Report",
        #     data=report_md,
        #     file_name="report.md",
        #     mime="text/markdown",
        # )
    # --- Save categorized CSV ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"transaction_details_{timestamp}.csv"

    output_dir = os.path.join(os.getcwd(), "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    df.to_csv(output_path, index=False)

    st.success(f"✅ Saved categorized file as {output_name}")
    st.download_button(
        label="Download categorized CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=output_name,
        mime="text/csv",
    )
else:
    st.info("Please upload one or more CSV files to begin.")

st.subheader("Model Evaluation")
if st.button(
    "Run evaluation on labeled data",
    help="Load labeled datasets from data/eval, recategorize them, and compare predictions against the provided ground truth.",
):
    eval_result = run_evaluation(batch_size=batch_size, preview_rows=25)
    summaries = eval_result.get("summaries", [])
    previews = eval_result.get("previews", {})
    all_results = eval_result.get("all_results")

    if not summaries:
        st.info("No evaluation files found or files are missing ground-truth columns.")
    else:
        if isinstance(all_results, pd.DataFrame) and not all_results.empty:
            summary_df = pd.DataFrame(summaries)
            st.write("Evaluation summary (percent match per evaluation file):")
            st.dataframe(summary_df)

            total_predictions = len(all_results)
            source_group = all_results.groupby("prediction_source", dropna=False)

            match_summary = source_group.agg(
                assignments=("prediction_source", "size"),
                exclude_match_pct=("exclude_flag_match", lambda s: round(float(s.mean() * 100), 2)),
                category_match_pct=("category_match", lambda s: round(float(s.mean() * 100), 2)),
                sub_category=("sub_category_match", lambda s: round(float(s.mean() * 100), 2)),
            ).reset_index()

            match_summary["assignment_pct"] = (
                match_summary["assignments"] / total_predictions * 100
            ).round(2)
            match_summary["category source"] = match_summary["prediction_source"].fillna("unknown")
            match_summary = match_summary[
                ["category source", "assignment_pct", "exclude_match_pct", "category_match_pct", "sub_category"]
            ].sort_values("assignment_pct", ascending=False)

            st.write("Prediction quality by category source:")
            st.dataframe(match_summary)

            eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_dir = os.path.join(os.getcwd(), "data", "eval")
            os.makedirs(eval_dir, exist_ok=True)
            eval_filename = f"eval_results_{eval_timestamp}.csv"
            eval_path = os.path.join(eval_dir, eval_filename)
            all_results.to_csv(eval_path, index=False)
            st.success(f"Saved detailed evaluation results to data/eval/{eval_filename}")
        else:
            summary_df = pd.DataFrame(summaries)
            st.write("Evaluation summary (percent match per evaluation file):")
            st.dataframe(summary_df)

        for summary in summaries:
            file_name = summary["file"]
            preview_df = previews.get(file_name)
            if preview_df is None or preview_df.empty:
                continue
            st.write(f"Preview for {file_name} (first 25 rows, mismatches first):")
            st.dataframe(preview_df)
