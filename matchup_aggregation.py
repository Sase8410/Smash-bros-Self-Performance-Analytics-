import pandas as pd

df = pd.read_csv("engineered_smash_dataset.csv")

matchups = (
    df.groupby(["character", "opponent_character"])
    .agg(
        matches=("match_id", "count"),

        win_rate=("result", "mean"),

        avg_damage_differential=(
            "damage_differential", "mean"
        ),

        avg_damage_ratio=(
            "damage_ratio", "mean"
        ),

        avg_ko_differential=(
            "ko_differential", "mean"
        ),

        avg_damage_per_minute=(
            "damage_per_minute", "mean"
        ),

        avg_damage_taken_per_minute=(
            "damage_taken_per_minute", "mean"
        ),

        avg_match_intensity=(
            "match_intensity", "mean"
        ),

        avg_match_time=(
            "match_time_seconds", "mean"
        ),

        avg_gsp=(
            "gsp", "mean"
        )
    )
    .reset_index()
)

matchups = matchups[matchups["matches"] >= 2].copy()

matchups.to_csv("matchup_features.csv",index=False)

print("Matchup aggregation complete.")
print(f"Total matchups: {len(matchups)}")

print("\nPreview:")
print(matchups.head())

print("\nDataset saved as: " "matchup_features.csv")