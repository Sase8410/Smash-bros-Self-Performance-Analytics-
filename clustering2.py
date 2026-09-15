# How do I play?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

df = pd.read_csv("engineered_smash_dataset.csv")

output_dir = "outputs/match_clustering"
os.makedirs(output_dir, exist_ok=True)

cluster_features = [
    "kos",
    "falls",
    "sds",
    "damage_given",
    "damage_taken",
    "match_time_seconds",
    "damage_per_minute",
    "damage_taken_per_minute",
    "match_intensity",
]

cluster_features = [c for c in cluster_features if c in df.columns]

X = df[cluster_features].copy()

X = X.replace([np.inf, -np.inf], np.nan)
if X.isnull().any().any():
    print("Warning: dropping rows with missing/infinite feature values:")
    print(X.isnull().sum()[X.isnull().sum() > 0])
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    df = df.loc[valid_idx].reset_index(drop=True)
    X = X.reset_index(drop=True)

zero_var_cols = [c for c in X.columns if X[c].std() == 0]
if zero_var_cols:
    print(f"Dropping zero-variance feature(s) (no info for clustering): {zero_var_cols}")
    X = X.drop(columns=zero_var_cols)
    cluster_features = [c for c in cluster_features if c not in zero_var_cols]

print(f"\nClustering on {len(X)} matches using features:")
print(cluster_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_range = range(2, 9)
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].set_xticks(list(k_range))

axes[1].plot(list(k_range), silhouette_scores, marker="o", color="#55A868")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(list(k_range))

plt.tight_layout()
plt.savefig(f"{output_dir}/k_selection.png", dpi=150)
plt.close()

best_k = list(k_range)[int(np.argmax(silhouette_scores))]
print(f"\nSilhouette scores by k: {dict(zip(k_range, [round(s, 3) for s in silhouette_scores]))}")
print(f"Best k by silhouette score: {best_k}")

K = best_k

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nFinal model: k={K}")
print(df["cluster"].value_counts().sort_index())

cluster_profile = df.groupby("cluster")[cluster_features].mean().round(2)
print("\nCluster profiles (mean values, original units):")
print(cluster_profile)
cluster_profile.to_csv(f"{output_dir}/cluster_profiles.csv")

profile_z = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()

fig_width = max(9, len(cluster_features) * 1.3)
plt.figure(figsize=(fig_width, max(3, K * 1.0)))
sns.heatmap(
    profile_z,
    annot=cluster_profile.values,
    fmt=".1f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Z-score vs other clusters", "shrink": 0.7},
)
plt.title("Cluster Profiles (color = relative z-score, text = actual value)")
plt.xlabel("Feature")
plt.ylabel("Cluster")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{output_dir}/cluster_profile_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

win_rate_by_cluster = df.groupby("cluster")["result"].mean().round(3)
print("\nWin rate by cluster (interpretation only, not used to build clusters):")
print(win_rate_by_cluster)

character_cluster_mix = pd.crosstab(
    df["character"], df["cluster"], normalize="index"
).round(2)
print("\nCharacter -> cluster mix (row-normalized):")
print(character_cluster_mix)

fig, ax1 = plt.subplots(figsize=(8, 5))
win_rate_by_cluster.plot(kind="bar", color="#4C72B0", ax=ax1)
ax1.axhline(df["result"].mean(), color="red", linestyle="--", label="Overall Win Rate")
ax1.set_title("Win Rate by Cluster")
ax1.set_xlabel("Cluster")
ax1.set_ylabel("Win Rate")
ax1.set_ylim(0, 1)
ax1.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/win_rate_by_cluster.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
character_cluster_mix.plot(kind="bar", stacked=True, colormap="tab10", figsize=(9, 5))
plt.title("Character Mix Within Each Cluster")
plt.xlabel("Character")
plt.ylabel("Share of Matches by Cluster")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{output_dir}/character_cluster_mix.png", dpi=150)
plt.close()

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=df["cluster"], cmap="tab10", alpha=0.75, s=50
)
plt.title("Match Clusters (PCA-projected to 2D)")
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.savefig(f"{output_dir}/clusters_pca.png", dpi=150)
plt.close()

df.to_csv("clustered_smash_dataset.csv", index=False)

print(f"\nClustering complete. k={K}")
print(f"Charts saved in: {output_dir}")
print("Clustered dataset saved to: clustered_smash_dataset.csv")