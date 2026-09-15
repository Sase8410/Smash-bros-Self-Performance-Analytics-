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
<img width="2700" height="1050" alt="image" src="https://github.com/user-attachments/assets/7d296120-8d10-4706-b5df-c545974fe8a1" />

**Clustering (match-level):** K-Means on individual matches that deliberately excludes results reflects how a match was played, not whether it was won. Found k = 2 via silhouette score: a slower playing style (117 matches, 64% win rate) and a faster, high-damage-pace style (35 matches, 51% win rate). Tempo is the dominant axis in the data.
<img width="1424" height="431" alt="image" src="https://github.com/user-attachments/assets/a3f27194-2941-4268-8e37-3b37bfa7f41b" />
<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/2ea86637-3c91-4ca1-ba4d-cd11459ae2fe" />

**Clustering (matchup-level):** A separate analysis on 23 aggregated (character, opponent) pairs. This one deliberately includes win_rate as a feature. Found k = 4, cleanly separating a 0% win rate, marked as the bad matchups cluster, from a 79% win rate, or the dominant matchups cluster. This answers a different question from the match-level clustering (matchup outcome vs. individual match style).
<img width="1332" height="581" alt="image" src="https://github.com/user-attachments/assets/5a1da3d5-2008-4328-b82a-5138ecf87d49" />
<img width="1200" height="900" alt="image" src="https://github.com/user-attachments/assets/768bcdf1-cb04-417f-ae88-97acfb23964d" />

**XGBoost:** Win/loss classifier reaches 73.7 cross-validated accuracy and 0.84 ROC-AUC after deliberately excluding kos, falls, ko_differential, and gsp_change, all of which correlate 0.8-0.9 with "result" column because they're essentially restatements of the outcome (Smash results are decided by KO count; GSP moves because of the result unless the opponent disconnects) rather than genuine predictors. damage_ratio (damage dealt relative to damage taken) is the strongest real signal.
<img width="1172" height="1169" alt="image" src="https://github.com/user-attachments/assets/44b7d25f-ec7a-447a-8069-0232909ff8a5" />
<img width="1200" height="720" alt="image" src="https://github.com/user-attachments/assets/7ff26890-1f1f-44cc-b566-7ac699b67441" />
<img width="750" height="600" alt="image" src="https://github.com/user-attachments/assets/a463d44d-3d8a-4969-ab94-ea9b013e0985" />
<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/434946b7-0e56-4c47-86ee-ab0f15e0eb3b" />


## Future improvements:
- More matches per character/matchup to stabilize the matchup-level clustering.
- Track additional results-screen stats if/when available for more characters

## Tech Stack
Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn
