import pandas as pd
import matplotlib.pyplot as plt

def summarize_spend(df, by="category"):
    if "amount_aud" in df.columns:
        value_column = "amount_aud"
    elif "amount" in df.columns:
        value_column = "amount"
    else:
        return pd.Series(dtype=float)
    summary = df.groupby(by)[value_column].sum().sort_values(ascending=False)
    return summary



def plot_spend(summary, title="Spending by Category"):
    if summary.empty:
        return None
    plt.figure(figsize=(8,4))
    summary.plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("spending_summary.png")
