"""
Trains a GRU-based sequence model to predict each customer's top spending category for 2026-02.

High-level flow:
- Loads the feature store (partitioned Parquet directory, single Parquet, or CSV).
- Aggregates transaction-level features to monthly customer-level features.
- Infers category spend columns and builds a supervised dataset where labels are
  the next month’s top category for each customer.
- Builds fixed-length monthly sequences per customer for training, testing, and 2026-02 prediction.
- Scales features using train-only statistics and wraps them in PyTorch datasets/dataloaders.
- Defines and trains a GRU classifier with a linear head and cross-entropy loss.
- Evaluates accuracy on a hold-out test month (2026-01) and then predicts
  the top category for 2026-02 for each customer.
- Saves predictions to 'data/gru_predictions_2026_02.csv' with one row per customer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import pyarrow.dataset as ds
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Ensure reproducible randomness across NumPy and PyTorch (CPU and GPU).
def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Convert arbitrary period-like strings to month start timestamps (e.g. '2025-08' → 2025-08-01).
def _to_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        bad = s[dt.isna()].head(5).tolist()
        raise ValueError(f"Found non-parseable period values (examples): {bad}")
    return dt.dt.to_period("M").dt.to_timestamp()


def _read_feature_store(path: Path) -> pd.DataFrame:
    """
    Reads either:
    - partitioned parquet dataset directory (e.g. data/feature_store/period=2025-08/*.parquet)
    - a single parquet file
    - a csv file
    """
    if path.is_dir():
        dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
        return dataset.to_table().to_pandas()

    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported feature store path: {path} (expected dir, .parquet, or .csv)")


# Infer which numeric columns represent spend by category, excluding known behavioral / payment features.
def _infer_category_columns(df: pd.DataFrame) -> List[str]:
    reserved = {
        "customer_id",
        "period",
        "month",
        "label_month",
        "target_category",
    }
    numeric_cols = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
    # Category columns are spend-by-category pivot outputs (strings in generate_data.py)
    # Heuristic: exclude behavioral & payment count features we know exist.
    known_non_category = {
        "total_spent",
        "txn_count",
        "avg_spent",
        "max_spent",
        "spend_std_dev",
        "online_ratio",
        "unique_merchants",
        "avg_spend_per_txn",
        "max_txn_ratio",
        "category_diversity",
        "prev_total_spent",
        "prev_txn_count",
        "prev_online_ratio",
        "spend_growth",
        "txn_growth",
        "online_ratio_change",
        # payment method pivots (counts)
        "card",
        "netbanking",
        "upi",
    }
    cats = [c for c in numeric_cols if c not in known_non_category]
    if not cats:
        raise ValueError("Could not infer category spend columns from feature store.")
    return sorted(cats)


# Aggregate raw period-level rows to customer–month level, summing counts/spend and averaging ratios.
def _aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = _to_month_start(df["period"])

    numeric_cols = [c for c in df.columns if c not in {"customer_id", "period", "month"} and pd.api.types.is_numeric_dtype(df[c])]

    # Sums for counts and spend-like columns; means for ratios/averages/std.
    sum_cols = set()
    mean_cols = set()
    for c in numeric_cols:
        lc = c.lower()
        if any(k in lc for k in ["ratio", "avg", "std", "change"]):
            mean_cols.add(c)
        else:
            sum_cols.add(c)

    agg_spec: Dict[str, str] = {}
    for c in sorted(sum_cols):
        agg_spec[c] = "sum"
    for c in sorted(mean_cols):
        agg_spec[c] = "mean"

    out = df.groupby(["customer_id", "month"], as_index=False).agg(agg_spec)
    out = out.sort_values(["customer_id", "month"]).reset_index(drop=True)
    return out


# Build feature (X) and label (y) frames where the label is the next month’s top spend category per customer.
def _make_supervised_frames(
    monthly: pd.DataFrame, category_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      X_frame: monthly features by customer+month
      y_frame: labels aligned to month t (label is top category at t+1), plus label_month
    """
    monthly = monthly.sort_values(["customer_id", "month"]).reset_index(drop=True)
    next_cats = monthly.groupby("customer_id")[category_cols].shift(-1)
    label_month = monthly.groupby("customer_id")["month"].shift(-1)

    y = pd.DataFrame(
        {
            "customer_id": monthly["customer_id"].values,
            "month": monthly["month"].values,  # feature month t
            "label_month": label_month.values,  # label month t+1
        }
    )
    # Avoid np.where evaluating both branches (nanargmax would error on all-NaN rows).
    next_arr = next_cats.to_numpy()
    # If a row is all-NaN (typically last available month for a customer), mark as invalid.
    valid = ~np.all(np.isnan(next_arr), axis=1)
    filled = np.where(np.isnan(next_arr), -np.inf, next_arr)
    argmax = filled.argmax(axis=1)
    target = np.full(shape=(len(y),), fill_value=None, dtype=object)
    target[valid] = np.array(category_cols, dtype=object)[argmax[valid]]
    y["target_category"] = target

    # Drop last month per customer (no next-month label)
    y = y.dropna(subset=["label_month", "target_category"]).reset_index(drop=True)

    X = monthly.copy()
    return X, y


@dataclass(frozen=True)
class SplitConfig:
    train_end_inclusive: pd.Timestamp  # label_month <= this is train
    test_label_month: pd.Timestamp  # exact label_month used for test metrics
    predict_month: pd.Timestamp  # month we forecast (no labels)


class CustomerSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,  # (N, L, F)
        labels: np.ndarray | None,  # (N,)
    ) -> None:
        self.x = torch.from_numpy(sequences).float()
        self.y = None if labels is None else torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # (B, L, H)
        last = out[:, -1, :]  # (B, H)
        return self.head(last)  # (B, C)


def _build_sequences_for_months(
    X_monthly: pd.DataFrame,
    y_frame: pd.DataFrame,
    feature_cols: List[str],
    category_cols: List[str],
    seq_len: int,
    split: SplitConfig,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],  # train (X, y)
    Tuple[np.ndarray, np.ndarray],  # test (X, y)
    Tuple[np.ndarray, List[str]],  # predict (X, customer_ids) for 2026-02
]:
    # Map category name -> class index
    cat_to_idx = {c: i for i, c in enumerate(category_cols)}

    # Join labels onto feature rows for month t
    labeled = y_frame.merge(
        X_monthly[["customer_id", "month"] + feature_cols],
        on=["customer_id", "month"],
        how="inner",
    )
    labeled["y"] = labeled["target_category"].map(cat_to_idx).astype(int)

    # Create per-customer month -> features lookup
    X_monthly = X_monthly.sort_values(["customer_id", "month"]).reset_index(drop=True)
    grouped = {cid: g for cid, g in X_monthly.groupby("customer_id", sort=False)}

    # Build a fixed-length monthly feature sequence ending at a given month for one customer.
    def make_seq(cid: str, end_month: pd.Timestamp) -> np.ndarray | None:
        g = grouped.get(cid)
        if g is None:
            return None
        g = g.set_index("month")
        months = pd.period_range(end_month.to_period("M") - (seq_len - 1), end_month.to_period("M"), freq="M").to_timestamp()
        try:
            seq = g.loc[months, feature_cols].to_numpy()
        except KeyError:
            return None
        if seq.shape != (seq_len, len(feature_cols)):
            return None
        return seq

    # Build train/test from labeled rows; use label_month for split (predicting that month)
    train_rows = labeled[labeled["label_month"] <= split.train_end_inclusive].copy()
    test_rows = labeled[labeled["label_month"] == split.test_label_month].copy()

    # Turn labeled rows into aligned (sequence, label) numpy arrays, skipping incomplete histories.
    def build_xy(rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for r in rows.itertuples(index=False):
            seq = make_seq(r.customer_id, r.month)
            if seq is None:
                continue
            xs.append(seq)
            ys.append(int(r.y))
        if not xs:
            raise ValueError("No sequences could be built. Try reducing --seq-len or check missing months per customer.")
        return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)

    X_train, y_train = build_xy(train_rows)
    X_test, y_test = build_xy(test_rows)

    # Build prediction sequences ending at last known month (2026-01) to predict 2026-02
    predict_end_month = split.predict_month - pd.offsets.MonthBegin(1)  # 2026-01-01
    pred_xs = []
    pred_cids = []
    for cid, g in grouped.items():
        if predict_end_month not in set(g["month"].values):
            continue
        seq = make_seq(cid, predict_end_month)
        if seq is None:
            continue
        pred_xs.append(seq)
        pred_cids.append(cid)
    if not pred_xs:
        raise ValueError("No prediction sequences built for 2026-02. Check if you have data up to 2026-01.")
    X_pred = np.stack(pred_xs, axis=0)
    return (X_train, y_train), (X_test, y_test), (X_pred, pred_cids)


# Compute simple classification accuracy from logits and integer class labels.
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


# End-to-end pipeline: load features, build sequences, train GRU, evaluate, and save 2026-02 predictions.
def train_and_predict(
    feature_store_path: Path,
    out_predictions_csv: Path,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> None:
    _set_seed(seed)

    raw = _read_feature_store(feature_store_path)
    if "customer_id" not in raw.columns or "period" not in raw.columns:
        raise ValueError("Feature store must contain at least customer_id and period columns.")

    monthly = _aggregate_to_monthly(raw)
    category_cols = _infer_category_columns(monthly)

    # Features: all numeric except category columns? We include them, because they help sequence modeling,
    # but the label uses next-month category amounts (shifted), so leakage isn't an issue (month t features).
    id_cols = {"customer_id", "month"}
    feature_cols = [c for c in monthly.columns if c not in id_cols and pd.api.types.is_numeric_dtype(monthly[c])]

    X_frame, y_frame = _make_supervised_frames(monthly, category_cols)

    # Forecast config: predict outcomes for 2026-02 using sequences ending 2026-01.
    split = SplitConfig(
        train_end_inclusive=pd.Timestamp("2025-12-01"),  # train labels up to 2025-12
        test_label_month=pd.Timestamp("2026-01-01"),  # evaluate predicting 2026-01
        predict_month=pd.Timestamp("2026-02-01"),
    )

    (X_train, y_train), (X_test, y_test), (X_pred, pred_cids) = _build_sequences_for_months(
        X_monthly=X_frame,
        y_frame=y_frame,
        feature_cols=feature_cols,
        category_cols=category_cols,
        seq_len=seq_len,
        split=split,
    )

    # Scale using train-only statistics
    scaler = StandardScaler()
    train_2d = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(train_2d)

    def scale_3d(x: np.ndarray) -> np.ndarray:
        x2 = x.reshape(-1, x.shape[-1])
        x2s = scaler.transform(x2)
        return x2s.reshape(x.shape[0], x.shape[1], x.shape[2]).astype(np.float32)

    X_train = scale_3d(X_train)
    X_test = scale_3d(X_test)
    X_pred = scale_3d(X_pred)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CustomerSequenceDataset(X_train, y_train)
    test_ds = CustomerSequenceDataset(X_test, y_test)
    pred_ds = CustomerSequenceDataset(X_pred, None)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    pred_dl = DataLoader(pred_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = GRUClassifier(
        input_dim=X_train.shape[-1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(category_cols),
        dropout=dropout,
    ).to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = torch.cuda.is_available()
    scaler_amp = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{epochs} [train]", leave=False)
        train_losses = []
        train_accs = []

        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                logits = model(xb)
                loss = crit(logits, yb)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()

            acc = _accuracy(logits.detach(), yb)
            train_losses.append(loss.item())
            train_accs.append(acc)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "batch_acc": f"{acc:.3f}"})

        model.eval()
        test_losses = []
        test_accs = []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = crit(logits, yb)
                test_losses.append(loss.item())
                test_accs.append(_accuracy(logits, yb))

        print(
            f"epoch {epoch}/{epochs} | "
            f"train_loss={np.mean(train_losses):.4f} train_acc={np.mean(train_accs):.3f} | "
            f"test_loss={np.mean(test_losses):.4f} test_acc={np.mean(test_accs):.3f}"
        )

    # Predict 2026-02 top category
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb in pred_dl:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    pred_idx = probs.argmax(axis=1)
    pred_cat = [category_cols[i] for i in pred_idx]

    out = pd.DataFrame({"customer_id": pred_cids, "predicted_top_category_2026_02": pred_cat})
    out_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_predictions_csv, index=False)
    print(f"Saved predictions to: {out_predictions_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature-store-path",
        type=Path,
        default=Path("data/feature_store"),
        help="Directory of partitioned parquet feature store (recommended), or a .parquet/.csv file.",
    )
    p.add_argument("--out-predictions-csv", type=Path, default=Path("data/gru_predictions_2026_02.csv"))
    p.add_argument("--seq-len", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_and_predict(
        feature_store_path=args.feature_store_path,
        out_predictions_csv=args.out_predictions_csv,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

