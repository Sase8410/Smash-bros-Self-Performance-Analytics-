import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_smash_dataset.csv")

# 1. Damage Differential
df["damage_differential"] = (
    df["damage_given"] - df["damage_taken"]
)

# 2. Damage Ratio
df["damage_ratio"] = (
    df["damage_given"] /
    df["damage_taken"].replace(0, np.nan)
)

# 3. KO Differential
df["ko_differential"] = (
    df["kos"] - df["falls"]
)

# 4. Damage Per Minute
if "match_time_seconds" in df.columns:

    df["damage_per_minute"] = (
        df["damage_given"] /
        (df["match_time_seconds"] / 60)
    )

# 5. Damage Taken Per Minute
if "match_time_seconds" in df.columns:

    df["damage_taken_per_minute"] = (
        df["damage_taken"] /
        (df["match_time_seconds"] / 60)
    )

# 6. GSP Change
df["gsp_change"] = (
    df.groupby("character")["gsp"]
    .diff()
)

df["gsp_change"] = df["gsp_change"].fillna(0)

# 7. Match Inensity
df["match_intensity"] = (
    (df["kos"] + df["falls"]) / df["match_time_seconds"].replace(0, np.nan)
)

df.to_csv(
    "engineered_smash_dataset.csv",
    index=False
)

print("Feature engineering complete.")
print("\nNew Features:")

print([
    "damage_differential",
    "damage_ratio",
    "ko_differential",
    "damage_per_minute",
    "damage_taken_per_minute",
    "gsp_change"
])

print(
    "\nDataset saved as: "
    "engineered_smash_dataset.csv"
)