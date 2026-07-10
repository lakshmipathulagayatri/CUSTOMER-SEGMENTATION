import pandas as pd
from sklearn.cluster import KMeans
import joblib

K = 4  # <-- set based on elbow/silhouette results from step 4

X_scaled = pd.read_csv("customer_data_scaled.csv")
df = pd.read_csv("customer_data.csv")

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

joblib.dump(kmeans, "kmeans_model.pkl")
df.to_csv("customer_data_clustered.csv", index=False)

print(f"Clustering complete with K={K}")
print(df["Cluster"].value_counts().sort_index())
print("\nSaved customer_data_clustered.csv and kmeans_model.pkl")