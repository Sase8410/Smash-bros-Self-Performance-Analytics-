import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("matchup_features.csv")

output_dir = "outputs/clustering"
os.makedirs(output_dir, exist_ok=True)

features = [
    "win_rate",
    "avg_damage_differential",
    "avg_damage_ratio",
    "avg_ko_differential",
    "avg_damage_per_minute",
    "avg_damage_taken_per_minute",
    "avg_match_intensity"
]

X = df[features].copy()

valid_rows = X.dropna().index

df = df.loc[valid_rows].reset_index(drop=True)
X = X.loc[valid_rows].reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Sizes:")

print(df["cluster"].value_counts().sort_index())

cluster_summary = (df.groupby("cluster")[features].mean().round(3))

print("\nCluster Summary:")
print(cluster_summary)

for cluster in sorted(df["cluster"].unique()):
    print(f"CLUSTER {cluster}")
    cluster_matchups = (
        df[df["cluster"] == cluster][
            [
                "character",
                "opponent_character",
                "matches",
                "win_rate",
                "avg_damage_differential",
                "avg_ko_differential"
            ]
        ]
        .sort_values(
            "win_rate",
            ascending=False
        )
    )

    print(cluster_matchups.to_string(index=False))

df.to_csv("clustered_matchups.csv",index=False)

cluster_summary.to_csv(f"{output_dir}/cluster_summary.csv")

print("\nClustered dataset saved as: ""clustered_matchups.csv")

print("Cluster summary saved as: ""outputs/clustering/cluster_summary.csv")