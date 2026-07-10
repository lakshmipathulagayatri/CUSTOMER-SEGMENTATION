import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("customer_data.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary stats:\n", df.describe())

# --- Distribution plots ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
num_cols = ["Age", "AnnualIncome", "SpendingScore",
            "PurchaseFrequency", "RecencyDays", "AvgOrderValue"]

for ax, col in zip(axes.flatten(), num_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution: {col}")

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150)
print("Saved eda_distributions.png")

# --- Correlation heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
print("Saved eda_correlation.png")

# --- Purchase pattern snapshot ---
print("\nAvg spending score by gender:\n", df.groupby("Gender")["SpendingScore"].mean())
print("\nAvg purchase frequency by age bracket:")
df["AgeBracket"] = pd.cut(df["Age"], bins=[17, 30, 45, 60, 80],
                           labels=["18-30", "31-45", "46-60", "61-80"])
print(df.groupby("AgeBracket", observed=True)["PurchaseFrequency"].mean())