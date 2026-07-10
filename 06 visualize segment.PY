import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv("customer_data_clustered.csv")
X_scaled = pd.read_csv("customer_data_scaled.csv")

# --- PCA scatter plot ---
pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = components[:, 0], components[:, 1]

plt.figure(figsize=(9, 7))
palette = sns.color_palette("Set2", n_colors=df["Cluster"].nunique())
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette=palette, s=60, alpha=0.8)
plt.title("Customer Segments (PCA-reduced 2D view)")
plt.savefig("segments_pca_scatter.png", dpi=150)
print("Saved segments_pca_scatter.png")
print(f"Explained variance by 2 PCA components: {pca.explained_variance_ratio_.sum():.2%}")

# --- Feature comparison across clusters ---
features = ["Age", "AnnualIncome", "SpendingScore",
            "PurchaseFrequency", "RecencyDays", "AvgOrderValue"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flatten(), features):
    sns.boxplot(data=df, x="Cluster", y=col, hue="Cluster", ax=ax, palette=palette, legend=False)
    ax.set_title(col)

plt.tight_layout()
plt.savefig("segments_feature_comparison.png", dpi=150)
print("Saved segments_feature_comparison.png")

# --- Cluster size pie chart ---
plt.figure(figsize=(6, 6))
df["Cluster"].value_counts().sort_index().plot.pie(
    autopct="%1.1f%%", colors=palette, ylabel=""
)
plt.title("Customer Segment Sizes")
plt.savefig("segments_pie.png", dpi=150)
print("Saved segments_pie.png")