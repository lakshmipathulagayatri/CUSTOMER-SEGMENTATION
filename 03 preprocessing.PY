import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

df = pd.read_csv("customer_data.csv")

# Encode Gender
le = LabelEncoder()
df["GenderEncoded"] = le.fit_transform(df["Gender"])  # Female=0, Male=1

features = ["Age", "GenderEncoded", "AnnualIncome", "SpendingScore",
            "PurchaseFrequency", "RecencyDays", "AvgOrderValue"]

X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=features)
X_scaled_df.to_csv("customer_data_scaled.csv", index=False)

# Persist for reuse (e.g. scoring new customers later)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "gender_encoder.pkl")

print("Scaled features saved to customer_data_scaled.csv")
print(X_scaled_df.head())