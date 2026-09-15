# Smash-bros-Self-Performance-Analytics-
A personal analytics pipeline that turns my own Super Smash Bros. Ultimate match history into a structured dataset, then applies unsupervised and supervised machine learning to answer two questions: how do I play, and what actually makes me win?

## Motivation
Every match I play produces a results screen full of stats (KOs, falls, damage dealt/taken, GSP, match time) that normally just get glanced at and forgotten. This project collects those stats systematically in a single unified dataset of all characters played and applies the same kind of analysis a data science team would apply to any structured tabular problem: cleaning, EDA, feature engineering, clustering and classification.

## Data
- 150+ matches logged across 6 characters
- Manually recorded stats directly from Smash's post-match results screen
- Raw columns: character, opponent_character, result, kos, falls, sds, damage_given, damage_taken, match_time, gsp

## Pipeline
Scripts run in this order:

1.	**data_cleaning.py:** cleaned_smash_dataset.csv
2.	**eda.py:** charts in outputs/eda/
3.	**feature_engineering.py:** engineered_smash_dataset.csv
4.	**matchup_aggregation.py:** matchup_features.csv (per-matchup aggregates, 2+ matches)
5a.	**clustering.py:** match-level playstyle clusters → clustered_smash_dataset.csv
5b.	**matchup_clustering.py:** matchup-level clusters → clustered_matchups.csv
6.	**xgboost_model.py:** win/loss classifier + feature importance / SHAP

## Key Findings
**EDA:** Overall win rate 61%. Win rate varies meaningfully by character, and matchup-level win rate varies far more than character-level win rate, which means who I'm fighting matters more than what I'm playing.
<img width="1200" height="750" alt="image" src="https://github.com/user-attachments/assets/b1332b7b-1b4a-4237-8d31-55334ebd672a" />


**Clustering (match-level):** K-Means on individual matches that deliberately excludes results reflects how a match was played, not whether it was won. Found k = 2 via silhouette score: a slower playing style (117 matches, 64% win rate) and a faster, high-damage-pace style (35 matches, 51% win rate). Tempo is the dominant axis in the data.

**Clustering (matchup-level):** A separate analysis on 23 aggregated (character, opponent) pairs. This one deliberately includes win_rate as a feature. Found k = 4, cleanly separating a 0% win rate, marked as the bad matchups cluster, from a 79% win rate, or the dominant matchups cluster. This answers a different question from the match-level clustering (matchup outcome vs. individual match style).

**XGBoost:** Win/loss classifier reaches 73.7 cross-validated accuracy and 0.84 ROC-AUC after deliberately excluding kos, falls, ko_differential, and gsp_change, all of which correlate 0.8-0.9 with "result" column because they're essentially restatements of the outcome (Smash results are decided by KO count; GSP moves because of the result unless the opponent disconnects) rather than genuine predictors. damage_ratio (damage dealt relative to damage taken) is the strongest real signal.

## Future improvements:
- More matches per character/matchup to stabilize the matchup-level clustering.
- Track additional results-screen stats if/when available for more characters

## Tech Stack
Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn
