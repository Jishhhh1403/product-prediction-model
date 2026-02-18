import pandas as pd
import numpy as np

np.random.seed(42)

customers = [f"C00{i}" for i in range(1,6)]
categories = ["groceries","electronics","travel",
              "fashion","fuel","entertainment"]
payment_methods = ["card","upi","netbanking"]
devices = ["mobile","web"]
cities = ["Mumbai","Delhi","Bangalore","Hyderabad"]

rows = []
dates = pd.date_range("2025-08-01","2026-01-31",freq="D")

txn_id = 1

for cust in customers:
    balance = np.random.randint(50000,200000)

    for date in dates:
        for _ in range(np.random.poisson(1.5)):
            amount = np.random.randint(100,20000)
            balance -= amount

            rows.append([
                cust,
                f"T{txn_id}",
                date,
                amount,
                np.random.choice(categories),
                f"M{np.random.randint(1,50)}",
                np.random.choice(payment_methods),
                np.random.choice(devices),
                np.random.choice(cities),
                balance,
                np.random.choice([0,1])
            ])
            txn_id += 1

df = pd.DataFrame(rows, columns=[
    "customer_id","transaction_id","timestamp",
    "amount","category","merchant_id",
    "payment_method","device","city",
    "balance","is_online"
])

df.to_csv("data/raw_transactions.csv", index=False)

print("Raw data generated")
