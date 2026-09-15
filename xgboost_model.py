import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, RocCurveDisplay
)
from xgboost import XGBClassifier

output_dir = "outputs/xgboost"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("engineered_smash_dataset.csv")

excluded_leakage = ["kos", "falls", "ko_differential", "gsp_change"]
excluded_zero_variance = ["sds"]

numeric_features = [
    "damage_given", "damage_taken", "damage_differential", "damage_ratio",
    "damage_per_minute", "damage_taken_per_minute", "match_intensity",
    "gsp", "match_time_seconds",
]
numeric_features = [c for c in numeric_features if c in df.columns]

print("Excluded as scoreboard-tautological / leakage:", excluded_leakage)
print("Excluded as zero-variance:", excluded_zero_variance)
print("Numeric features used:", numeric_features)

character_dummies = pd.get_dummies(df["character"], prefix="char")

opponent_familiarity = df.groupby("opponent_character")["opponent_character"].transform("count")
opponent_familiarity.name = "opponent_familiarity"

X = pd.concat([df[numeric_features], character_dummies, opponent_familiarity], axis=1)
y = df["result"]

print(f"\nFinal feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_model = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    eval_metric="logloss", random_state=42,
)
cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy")
print(f"\n5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(f"Per-fold: {[round(s, 3) for s in cv_scores]}")

model = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    eval_metric="logloss", random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nHoldout test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Holdout test ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification report (holdout):")
print(classification_report(y_test, y_pred, target_names=["Loss", "Win"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"],
)
plt.title("Confusion Matrix (Holdout Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="XGBoost")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax.set_title("ROC Curve (Holdout Test Set)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/roc_curve.png", dpi=150)
plt.close()

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv(f"{output_dir}/feature_importance.csv", header=["importance"])

plt.figure(figsize=(8, max(4, len(importances) * 0.3)))
importances.sort_values().plot(kind="barh", color="#4C72B0")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance (gain)")
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance.png", dpi=150)
plt.close()

print("\nTop features by importance:")
print(importances.head(10))

try:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSHAP summary saved to {output_dir}/shap_summary.png")
except ImportError:
    print("\nshap not installed -- skipping SHAP summary plot (pip install shap to enable).")

print(f"\nXGBoost analysis complete. Charts saved in: {output_dir}")