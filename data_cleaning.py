import pandas as pd

df = pd.read_csv('Smash bros games.csv')

df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_"))

text_columns = ["character", "opponent_character"]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].str.strip().str.lower()

df = df.drop_duplicates()

print("Missing values:")
print(df.isnull().sum())

if "match_id" in df.columns:
    duplicate_match_ids = df[df["match_id"].duplicated(keep=False)]

    print ("\nDuplicate match_id values:")
    print(duplicate_match_ids)

invalid_results = df[
    ((df["result"] == 1) & (df["kos"] <= df["falls"])) |
    ((df["result"]== 0) & (df["kos"] >= df["falls"]))
]

print("\nInvalid result rows:")
print(invalid_results)

numeric_columns = ["kos", "falls", "sds", "damage_given", "damage_taken", "gsp"]

for col in numeric_columns:
    if col in df.columns:
        invalid_negative = df[df[col] < 0]

        if not invalid_negative.empty:
            print(f"\nInvalid negative values in column '{col}':")
            print(invalid_negative)

invalid_result_values = df[~df["result"].isin([0, 1])]
print("\nInvalid result values:")
print(invalid_result_values)


if "match_time" in df.columns:
    time_parts = df["match_time"].str.split(":", expand=True)

    minutes = pd.to_numeric(time_parts[0])
    seconds = pd.to_numeric(time_parts[1])

    df["match_time_seconds"] = (minutes * 60) + seconds

    print("\nMatch time summary (seconds):")
    print(df["match_time_seconds"].describe())



print("\nDataset shape:")
print(df.shape)

print("\nUnique characters:")
print(df["character"].unique())

print(df)