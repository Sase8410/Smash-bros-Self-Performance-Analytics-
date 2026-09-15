# What kind of matchups is this?


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

output_dir = "outputs/matchup_clustering"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("matchup_features.csv")

features = [
    "win_rate",
    "avg_damage_differential",
    "avg_damage_ratio",
    "avg_ko_differential",
    "avg_damage_per_minute",
    "avg_damage_taken_per_minute",
    "avg_match_intensity",
]
features = [f for f in features if f in df.columns]

X = df[features].copy()

print("Missing values:")
print(X.isnull().sum())

X = X.replace([np.inf, -np.inf], np.nan)
if X.isnull().any().any():
    valid_idx = X.dropna().index
    print(f"\nDropping {len(X) - len(valid_idx)} row(s) with missing/infinite values.")
    df = df.loc[valid_idx].reset_index(drop=True)
    X = X.loc[valid_idx].reset_index(drop=True)

zero_var_cols = [c for c in X.columns if X[c].std() == 0]
if zero_var_cols:
    print(f"Dropping zero-variance feature(s): {zero_var_cols}")
    X = X.drop(columns=zero_var_cols)
    features = [f for f in features if f not in zero_var_cols]

print(f"\nClustering on {len(X)} matchups using features:")
print(features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_range = range(2, 8)
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow Method (Matchup-Level)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].set_xticks(list(k_range))

axes[1].plot(list(k_range), silhouette_scores, marker="o", color="#55A868")
axes[1].set_title("Silhouette Score by k (Matchup-Level)")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(list(k_range))

plt.tight_layout()
plt.savefig(f"{output_dir}/k_selection.png", dpi=150)
plt.close()

eval_table = pd.DataFrame({
    "k": list(k_range), "inertia": inertias, "silhouette_score": silhouette_scores
})
eval_table.to_csv(f"{output_dir}/k_evaluation.csv", index=False)

print(f"\nSilhouette scores by k: {dict(zip(k_range, [round(s, 3) for s in silhouette_scores]))}")

best_k = list(k_range)[int(np.argmax(silhouette_scores))]
print(f"Best k by silhouette score: {best_k}")

K = best_k

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nFinal model: k={K}")
print(df["cluster"].value_counts().sort_index())

cluster_profile = df.groupby("cluster")[features].mean().round(3)
print("\nCluster profiles (mean values, original units):")
print(cluster_profile)
cluster_profile.to_csv(f"{output_dir}/matchup_cluster_profiles.csv")

profile_z = (cluster_profile - cluster_profile.mean()) / cluster_profile.std()

fig_width = max(9, len(features) * 1.4)
plt.figure(figsize=(fig_width, max(3, K * 1.0)))
sns.heatmap(
    profile_z,
    annot=cluster_profile.values,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Z-score vs other clusters", "shrink": 0.7},
)
plt.title("Matchup Cluster Profiles (color = relative z-score, text = actual value)")
plt.xlabel("Feature")
plt.ylabel("Cluster")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{output_dir}/matchup_cluster_profile_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

for cluster in sorted(df["cluster"].unique()):
    print(f"\nCLUSTER {cluster}")
    cluster_matchups = (
        df[df["cluster"] == cluster][
            ["character", "opponent_character", "matches", "win_rate",
             "avg_damage_differential", "avg_ko_differential"]
        ]
        .sort_values("win_rate", ascending=False)
    )
    print(cluster_matchups.to_string(index=False))

opponent_cluster_mix = pd.crosstab(df["opponent_character"], df["cluster"])
opponent_cluster_mix.to_csv(f"{output_dir}/opponent_cluster_mix.csv")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=df["cluster"], cmap="tab10", alpha=0.8, s=70
)
for i, row in df.iterrows():
    plt.annotate(
        f"{row['character']} vs {row['opponent_character']}",
        (X_pca[i, 0], X_pca[i, 1]),
        fontsize=6, alpha=0.7,
    )
plt.title("Matchup Clusters (PCA-projected to 2D)")
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.savefig(f"{output_dir}/matchup_clusters_pca.png", dpi=150)
plt.close()

df.to_csv("clustered_matchups.csv", index=False)

print(f"\nMatchup-level clustering complete. k={K}")
print(f"Charts saved in: {output_dir}")
print("Clustered dataset saved to: clustered_matchups.csv")