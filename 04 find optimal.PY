import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_scaled = pd.read_csv("customer_data_scaled.csv")

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(list(k_range), silhouette_scores, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score by K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig("optimal_k_analysis.png", dpi=150)
print("Saved optimal_k_analysis.png")

best_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
print(f"\nSuggested K by best silhouette score: {best_k}")
for k, s in zip(k_range, silhouette_scores):
    print(f"K={k}: silhouette={s:.3f}")