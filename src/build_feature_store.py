import pandas as pd

df = pd.read_csv("data/raw_transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Monthly window
df["period"] = df["timestamp"].dt.to_period("M")

# ===============================
# CORE BEHAVIORAL FEATURES
# ===============================
agg = df.groupby(["customer_id", "period"]).agg(
    total_spent=("amount", "sum"),
    txn_count=("amount", "count"),
    avg_spent=("amount", "mean"),
    max_spent=("amount", "max"),
    spend_std_dev=("amount", "std"),   # NEW
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

# ===============================
# MERGE ALL FEATURES
# ===============================
feature_store = (
    agg.merge(cat_features, on=["customer_id", "period"])
       .merge(pay_features, on=["customer_id", "period"])
)

# ===============================
# TREND FEATURES (MONTH-TO-MONTH)
# ===============================

feature_store = feature_store.sort_values(["customer_id", "period"])

feature_store["prev_total_spent"] = feature_store.groupby("customer_id")["total_spent"].shift(1)
feature_store["prev_txn_count"] = feature_store.groupby("customer_id")["txn_count"].shift(1)
feature_store["prev_online_ratio"] = feature_store.groupby("customer_id")["online_ratio"].shift(1)

# Growth features
feature_store["spend_growth"] = feature_store["total_spent"] - feature_store["prev_total_spent"]
feature_store["txn_growth"] = feature_store["txn_count"] - feature_store["prev_txn_count"]
feature_store["online_ratio_change"] = feature_store["online_ratio"] - feature_store["prev_online_ratio"]

# Replace NaNs (first month)
feature_store.fillna(0, inplace=True)

# ===============================
# SAVE FEATURE STORE
# ===============================

feature_store["period"] = feature_store["period"].astype(str)

feature_store.to_parquet(
    "data/feature_store/",
    engine="pyarrow",
    partition_cols=["period"],
    index=False,
)

print("Feature store created with behavioral + trend features")
