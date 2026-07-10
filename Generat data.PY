import numpy as np
import pandas as pd

np.random.seed(42)
n_customers = 1000

def make_segment(n, age_range, income_range, spend_range, freq_range, recency_range, aov_range, label):
    return pd.DataFrame({
        "Age": np.random.randint(*age_range, n),
        "AnnualIncome": np.random.randint(*income_range, n),
        "SpendingScore": np.random.randint(*spend_range, n),          # 1-100 scale
        "PurchaseFrequency": np.random.randint(*freq_range, n),       # orders per year
        "RecencyDays": np.random.randint(*recency_range, n),          # days since last purchase
        "AvgOrderValue": np.random.uniform(*aov_range, n).round(2),
        "TrueSegment": label
    })

segments = pd.concat([
    make_segment(250, (18, 30), (20000, 45000), (60, 100), (15, 30), (1, 20),   (15, 60),  "Young High-Engagement"),
    make_segment(250, (35, 55), (70000, 150000), (60, 95), (10, 25), (1, 30),   (80, 300), "Affluent Loyal"),
    make_segment(250, (25, 45), (30000, 60000), (10, 40), (1, 6),  (90, 365), (20, 70),  "Budget Infrequent"),
    make_segment(250, (45, 70), (40000, 90000), (30, 60), (4, 12), (30, 120), (40, 150), "Occasional Mid-Value"),
], ignore_index=True)

segments["CustomerID"] = ["CUST" + str(i).zfill(5) for i in range(1, len(segments) + 1)]
segments["Gender"] = np.random.choice(["Male", "Female"], size=len(segments))

# shuffle rows so segments aren't pre-sorted (more realistic)
df = segments.sample(frac=1, random_state=42).reset_index(drop=True)

cols = ["CustomerID", "Age", "Gender", "AnnualIncome", "SpendingScore",
        "PurchaseFrequency", "RecencyDays", "AvgOrderValue", "TrueSegment"]
df = df[cols]

df.to_csv("customer_data.csv", index=False)
print(f"Saved {len(df)} rows to customer_data.csv")
print(df.head())