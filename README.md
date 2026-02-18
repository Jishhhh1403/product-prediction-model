# Product Prediction (GRU) — 2026-02 Top Spend Category

Train a **GRU-based sequence classifier** to predict each customer’s **top spending category for 2026-02**, using monthly customer features built from transaction history.

The included scripts can also generate **synthetic** transaction data so you can run the full pipeline end-to-end.

## What it does

- Builds a **monthly feature store** from transaction-level data (customer-month aggregates + category spend pivots + payment-method counts + simple trends).
- Creates a supervised dataset where the label at month \(t\) is the **top category by spend at month \(t+1\)**.
- Trains a **PyTorch GRU** on fixed-length monthly sequences.
- Evaluates on **2026-01** (predicting 2026-01 from prior months) and then generates predictions for **2026-02**.
- Writes one row per customer to `data/gru_predictions_2026_02.csv`.

## Project structure

- `src/generate_data.py`: generates `data/raw_transactions.csv` (synthetic transactions from 2025-08-01 to 2026-01-31)
- `src/build_feature_store.py`: builds a partitioned Parquet feature store at `data/feature_store/` (partitioned by `period=YYYY-MM`)
- `src/clean_feature_store_partitions.py`: deletes existing `data/feature_store/period=*` partitions (safe rebuild helper)
- `src/train_gru_predict_2026_02.py`: trains + evaluates + predicts 2026-02 and writes `data/gru_predictions_2026_02.csv`

## Setup (Windows / PowerShell)

From the repo root:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Quickstart (end-to-end)

Run these from the repo root:

```bash
# 1) Generate synthetic transaction data
python .\src\generate_data.py

# 2) (Optional) Clean existing parquet partitions
python .\src\clean_feature_store_partitions.py

# 3) Build monthly feature store (partitioned parquet)
python .\src\build_feature_store.py

# 4) Train GRU + evaluate on 2026-01 + predict 2026-02
python .\src\train_gru_predict_2026_02.py
```

Output:

- `data/gru_predictions_2026_02.csv`
  - Columns: `customer_id`, `predicted_top_category_2026_02`

## Training / prediction script usage

The main entrypoint is `src/train_gru_predict_2026_02.py`.

By default it reads the feature store from `data/feature_store/` (recommended). You can also point it at a single `.parquet` file or a `.csv`.

Example (custom hyperparameters):

```bash
python .\src\train_gru_predict_2026_02.py `
  --feature-store-path .\data\feature_store `
  --out-predictions-csv .\data\gru_predictions_2026_02.csv `
  --seq-len 3 `
  --hidden-dim 64 `
  --num-layers 1 `
  --dropout 0.1 `
  --batch-size 64 `
  --epochs 10 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --seed 42
```

Notes:

- **Sequence length**: `--seq-len` is the number of prior months used to predict the next month’s top category. If you see errors about missing sequences, try reducing this value.
- **GPU**: if CUDA is available, the script will use it automatically and train with AMP.
- **Train/test/predict months are fixed in code**:
  - Train labels through `2025-12`
  - Test label month `2026-01`
  - Predict month `2026-02` (using sequences ending `2026-01`)

## Data expectations

Feature store input (dir / parquet / csv) must include at least:

- `customer_id`
- `period` (parseable as a month; e.g. `2025-08`, `2025-08-01`, etc.)

Category spend columns are inferred heuristically from numeric columns (excluding known behavioral/trend/payment columns). The model predicts one of these inferred categories.

## Troubleshooting

- **“No sequences could be built…”**: some customers may not have a full contiguous history for the chosen `--seq-len`. Reduce `--seq-len`, or ensure each customer has rows for each month in the lookback window.
- **“No prediction sequences built for 2026-02…”**: ensure your feature store contains data up to **2026-01** (the script predicts 2026-02 from sequences ending in 2026-01).

## Dependencies

See `requirements.txt`:

- pandas, numpy, pyarrow
- torch
- scikit-learn
- tqdm

