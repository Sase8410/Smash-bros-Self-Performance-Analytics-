import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("matchup_features.csv")

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

print("Missing values:")
print(X.isnull().sum())

valid_rows = X.dropna().index

df = df.loc[valid_rows].reset_index(drop=True)
X = X.loc[valid_rows].reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []

k_values = range(2, 8)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Elbow Method Plot
plt.figure(figsize=(7, 5))

plt.plot(k_values, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.tight_layout()
plt.savefig("outputs/elbow_method.png", dpi=150)
plt.close()

# Silhouette Score Plot
plt.figure(figsize=(7, 5))

plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")

plt.tight_layout()
plt.savefig("outputs/silhouette_scores.png", dpi=150)
plt.close()

print("\nClustering Evaluation:")
for k, inertia, score in zip(k_values, inertias, silhouette_scores):
    print(f"k={k}: Inertia={inertia:.2f}, Silhouette Score={score:.3f}")

best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]

print(f"\nHighest silhouette score: k = {best_k}")