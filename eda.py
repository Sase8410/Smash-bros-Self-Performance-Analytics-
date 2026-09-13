import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_smash_dataset.csv")

output_dir = "outputs/eda"
os.makedirs(output_dir, exist_ok=True)

print("Super Smash Bros EDA\n")

# 1. Overall Summary
total_matches = len(df)
total_wins = df["result"].sum()
total_losses = total_matches - total_wins
overall_win_rate = (total_wins / total_matches) * 100

print(f"Total Matches: {total_matches}")
print(f"Wins: {total_wins}")
print(f"Losses: {total_losses}")
print(f"Win Rate: {overall_win_rate:.2f}%")
print(f"Average KOs: {df['kos'].mean():.2f}")
print(f"Average Falls: {df['falls'].mean():.2f}")
print(f"Average Damage Given: {df['damage_given'].mean():.2f}")
print(f"Average Damage Taken: {df['damage_taken'].mean():.2f}")
print(f"Average GSP: {df['gsp'].mean():,.0f}")

if "match_time_seconds" in df.columns:
    avg_time = df["match_time_seconds"].mean()
    print(
        f"Average Match Time: "
        f"{int(avg_time // 60)}:{int(avg_time % 60):02d}"
    )


# 2. Win Rate by Character
win_rate_by_character = (
    df.groupby("character")["result"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

win_rate_by_character.plot(
    kind="bar",
    color="#4C72B0"
)

plt.axhline(
    df["result"].mean(),
    color="red",
    linestyle="--",
    label="Overall Win Rate"
)

plt.title("Win Rate by Character")
plt.xlabel("Character")
plt.ylabel("Win Rate")
plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{output_dir}/win_rate_by_character.png",
    dpi=150
)

plt.close()

# 3. Damage Given vs Damage Taken

plt.figure(figsize=(7, 6))

sns.scatterplot(
    data=df,
    x="damage_taken",
    y="damage_given",
    hue="result",
    palette={0: "#C44E52", 1: "#55A868"},
    alpha=0.75
)

minimum = df[
    ["damage_given", "damage_taken"]
].min().min()

maximum = df[
    ["damage_given", "damage_taken"]
].max().max()

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    color="gray"
)

plt.title("Damage Given vs Damage Taken")
plt.xlabel("Damage Taken")
plt.ylabel("Damage Given")
plt.legend(
    title="Result",
    labels=["Loss", "Win"]
)

plt.tight_layout()

plt.savefig(
    f"{output_dir}/damage_given_vs_taken.png",
    dpi=150
)

plt.close()


# 4. KOs by Result
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="result",
    y="kos",
    hue="result",
    palette={0: "#C44E52", 1: "#55A868"},
    legend=False
)

plt.xticks([0, 1], ["Loss", "Win"])
plt.title("KOs by Match Result")
plt.xlabel("Result")
plt.ylabel("KOs")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/kos_by_result.png",
    dpi=150
)

plt.close()

# 5. Falls by Result
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="result",
    y="falls",
    hue="result",
    palette={0: "#C44E52", 1: "#55A868"},
    legend=False
)

plt.xticks([0, 1], ["Loss", "Win"])
plt.title("Falls by Match Result")
plt.xlabel("Result")
plt.ylabel("Falls")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/falls_by_result.png",
    dpi=150
)

plt.close()


# 6. Average GSP by Character
gsp_by_character = (
    df.groupby("character")["gsp"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

gsp_by_character.plot(
    kind="bar",
    color="#4C72B0"
)

plt.title("Average GSP by Character")
plt.xlabel("Character")
plt.ylabel("Average GSP")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/gsp_by_character.png",
    dpi=150
)

plt.close()

# 7. Average Match Time by Character
if "match_time_seconds" in df.columns:

    match_time_by_character = (
        df.groupby("character")["match_time_seconds"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(8, 5))

    match_time_by_character.plot(
        kind="bar",
        color="#8172B3"
    )

    plt.title("Average Match Time by Character")
    plt.xlabel("Character")
    plt.ylabel("Average Match Time (seconds)")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/match_time_by_character.png",
        dpi=150
    )

    plt.close()

# 8. Matchup Win Rate Heatmap
matchup_counts = (
    df.groupby(["character", "opponent_character"])
    .size()
    .reset_index(name="matches")
)

valid_matchups = matchup_counts[
    matchup_counts["matches"] >= 2
]

filtered_matchups = df.merge(
    valid_matchups[["character", "opponent_character"]],
    on=["character", "opponent_character"],
    how="inner"
)

matchup_win_rate = filtered_matchups.pivot_table(
    index="character",
    columns="opponent_character",
    values="result",
    aggfunc="mean"
)

plt.figure(figsize=(18, 7))

sns.heatmap(
    matchup_win_rate,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    linewidths=0.5
)

plt.title("Win Rate by Character Matchup (2+ Matches)")
plt.xlabel("Opponent Character")
plt.ylabel("My Character")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/matchup_win_rate_heatmap.png",
    dpi=150
)

plt.close()

# 9. Correlation Heatmap
numeric_cols = [
    "result",
    "kos",
    "falls",
    "sds",
    "damage_given",
    "damage_taken",
    "gsp"
]

if "match_time_seconds" in df.columns:
    numeric_cols.append("match_time_seconds")

corr = df[numeric_cols].corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/correlation_heatmap.png",
    dpi=150
)

plt.close()

print("\nEDA complete.")
print(f"Charts saved in: {output_dir}")