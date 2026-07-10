

import pandas as pd

df = pd.read_csv("customer_data_clustered.csv")

features = ["Age", "AnnualIncome", "SpendingScore",
            "PurchaseFrequency", "RecencyDays", "AvgOrderValue"]

profile = df.groupby("Cluster")[features].mean().round(1)
profile["CustomerCount"] = df.groupby("Cluster").size()
profile["PctOfBase"] = (profile["CustomerCount"] / len(df) * 100).round(1)

print("=== Segment Profile Table ===")
print(profile)
profile.to_csv("segment_profiles.csv")
print("\nSaved segment_profiles.csv")


def describe_segment(row):
    tags = []
    tags.append("young" if row["Age"] < 32 else "middle-aged" if row["Age"] < 50 else "older")
    tags.append("high income" if row["AnnualIncome"] > 70000 else "moderate income" if row["AnnualIncome"] > 40000 else "budget-conscious")
    tags.append("highly engaged" if row["SpendingScore"] > 60 else "low engagement")
    tags.append("frequent buyer" if row["PurchaseFrequency"] > 20 else "occasional buyer" if row["PurchaseFrequency"] > 8 else "rare buyer")
    tags.append("recently active" if row["RecencyDays"] < 30 else "at risk of churn" if row["RecencyDays"] > 90 else "moderately active")
    return ", ".join(tags)


def recommend_action(row):
    if row["SpendingScore"] > 60 and row["AnnualIncome"] > 70000:
        return "VIP treatment: early access, premium loyalty tier, upsell high-margin products."
    if row["RecencyDays"] > 90:
        return "Win-back campaign: discount code, re-engagement email series."
    if row["PurchaseFrequency"] > 20 and row["AnnualIncome"] < 45000:
        return "Loyalty rewards & bundle deals to protect high-frequency, price-sensitive shoppers."
    return "Nurture campaign: educational content, mid-tier promotions to increase frequency."


print("\n=== Segment Insights ===")
for cluster_id, row in profile.iterrows():
    print(f"\nCluster {cluster_id} ({int(row['CustomerCount'])} customers, {row['PctOfBase']}% of base)")
    print(f"  Profile: {describe_segment(row)}")
    print(f"  Recommended action: {recommend_action(row)}")