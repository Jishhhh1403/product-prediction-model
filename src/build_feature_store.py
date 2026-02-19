"""
Builds a monthly, customer-level feature store from raw transaction data.

- Reads 'data/raw_transactions.csv' and converts timestamps to monthly periods.
- Aggregates core behavioral metrics (spend, counts, averages, variability, online ratio, unique merchants).
- Creates category-level spend pivots and a category diversity feature.
- Builds payment-method count features using a pivot on 'payment_method'.
- Adds month-to-month trend features (spend and transaction growth, online ratio change).
- Fills missing values and writes a partitioned Parquet feature store under 'data/feature_store/period=...'.
"""

import pandas as pd
from pathlib import Path
import shutil

df = pd.read_csv("data/raw_transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Monthly window: aggregate all features at customer–month granularity.
df["period"] = df["timestamp"].dt.to_period("M")

# ===============================
# CORE BEHAVIORAL FEATURES
# ===============================
agg = df.groupby(["customer_id", "period"]).agg(
    total_spent=("amount", "sum"),
    txn_count=("amount", "count"),
    avg_spent=("amount", "mean"),
    min_spent=("amount", "min"),
    max_spent=("amount", "max"),
    spend_std_dev=("amount", "std"),   
    online_ratio=("is_online", "mean"),
    unique_merchants=("merchant_id", "nunique"),
).reset_index()

# Replace NaN std with 0
agg["spend_std_dev"] = agg["spend_std_dev"].fillna(0)

# Derived behavioral features
agg["avg_spend_per_txn"] = agg["total_spent"] / agg["txn_count"]
agg["max_txn_ratio"] = agg["max_spent"] / agg["total_spent"]

# ===============================
# CATEGORY FEATURES
# ===============================
cat_features = pd.pivot_table(
    df,
    values="amount",
    index=["customer_id", "period"],
    columns="category",
    aggfunc="sum",
    fill_value=0,
).reset_index()

# Category diversity
category_cols = cat_features.columns.difference(["customer_id", "period"])
cat_features["category_diversity"] = (cat_features[category_cols] > 0).sum(axis=1)

# ===============================
# PAYMENT FEATURES
# ===============================
pay_features = pd.pivot_table(
    df,
    values="amount",
    index=["customer_id", "period"],
    columns="payment_method",
    aggfunc="count",
    fill_value=0,
).reset_index()
payment_cols = pay_features.columns.difference(["customer_id", "period"])

# ===============================
def _write_partitioned(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    out["period"] = out["period"].astype(str)

    out_path = Path(path)
    if out_path.exists():
        shutil.rmtree(out_path)

    out.to_parquet(
        out_path,
        engine="pyarrow",
        partition_cols=["period"],
        index=False,
    )


# ===============================
# MERGE ALL FEATURES (COMBINED VIEW)
# ===============================
feature_store = (
    agg.merge(cat_features, on=["customer_id", "period"])
    .merge(pay_features, on=["customer_id", "period"])
)

# ===============================
# TREND & GROWTH FEATURES (MONTH-TO-MONTH)
# ===============================
feature_store = feature_store.sort_values(["customer_id", "period"])

feature_store["prev_total_spent"] = feature_store.groupby("customer_id")["total_spent"].shift(1)
feature_store["prev_txn_count"] = feature_store.groupby("customer_id")["txn_count"].shift(1)
feature_store["prev_online_ratio"] = feature_store.groupby("customer_id")["online_ratio"].shift(1)

feature_store["spend_growth"] = feature_store["total_spent"] - feature_store["prev_total_spent"]
feature_store["txn_growth"] = feature_store["txn_count"] - feature_store["prev_txn_count"]
feature_store["online_ratio_change"] = feature_store["online_ratio"] - feature_store["prev_online_ratio"]

# Replace NaNs (first month)
feature_store.fillna(0, inplace=True)

# ===============================
# SPLIT INTO GRU-ORIENTED FEATURE TABLES
# ===============================
core_behavior_cols = [
    "total_spent",
    "txn_count",
    "avg_spent",
    "min_spent",
    "max_spent",
    "spend_std_dev",
    "online_ratio",
    "unique_merchants",
    "avg_spend_per_txn",
    "max_txn_ratio",
]

core_behavior_features = feature_store[["customer_id", "period"] + core_behavior_cols].copy()
category_features = feature_store[["customer_id", "period"] + list(category_cols) + ["category_diversity"]].copy()
payment_features = feature_store[["customer_id", "period"] + list(payment_cols)].copy()
growth_features = feature_store[["customer_id", "period", "spend_growth", "txn_growth"]].copy()
trend_features = feature_store[
    [
        "customer_id",
        "period",
        "prev_total_spent",
        "prev_txn_count",
        "prev_online_ratio",
        "online_ratio_change",
    ]
].copy()

# ===============================
# WRITE OUT PARTITIONED FEATURE TABLES
# ===============================
_write_partitioned(core_behavior_features, "data/feature_store_core_behavior")
_write_partitioned(category_features, "data/feature_store_category")
_write_partitioned(payment_features, "data/feature_store_payment")
_write_partitioned(growth_features, "data/feature_store_growth")
_write_partitioned(trend_features, "data/feature_store_trend")

# Keep original combined feature store for existing GRU pipeline
_write_partitioned(feature_store, "data/feature_store")

print(
    "Feature store created with separate tables: "
    "core_behavior, category, payment, growth, trend (plus combined view)"
)
