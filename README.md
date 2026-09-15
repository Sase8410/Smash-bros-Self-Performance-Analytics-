# Super Smash Bros. Self-Performance Analytics

An end-to-end data science and machine learning project analyzing my personal performance in **Super Smash Bros. Ultimate** using self-collected match data.

Rather than relying on an existing public dataset, I recorded statistics from my own matches and built a complete analytics pipeline to answer three primary questions:

1. **Which characters do I perform best with?**
2. **Which matchups do I dominate or struggle against?**
3. **What gameplay patterns and statistics are most associated with winning?**

The project combines **data cleaning, exploratory data analysis, feature engineering, K-Means clustering, PCA, XGBoost classification, and SHAP model interpretation** to transform raw match statistics into actionable insights about my gameplay.

---

# Project Overview

Each observation in the original dataset represents one online match and contains information about the character I played, the opponent character, the match result, and several gameplay statistics.

The complete pipeline is:

```text
Raw Match Data
      ↓
Data Cleaning & Validation
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
      ├───────────────┐
      ↓               ↓
Match-Level       Matchup-Level
Clustering         Aggregation
      ↓               ↓
K-Means           K-Means
      ↓               ↓
PCA               PCA
      └───────┬───────┘
              ↓
      XGBoost Classification
              ↓
       SHAP Interpretation
              ↓
         Final Insights
```

The project therefore approaches performance from both an **unsupervised learning perspective**, where natural gameplay patterns are discovered without using predefined labels, and a **supervised learning perspective**, where match outcomes are predicted from gameplay statistics.

---

# Dataset

The dataset contains **152 self-collected Super Smash Bros. Ultimate matches** across six characters:

- Mario
- Toon Link
- Lucas
- Captain Falcon
- Pit
- Banjo & Kazooie

The primary raw variables include:

| Feature | Description |
|---|---|
| `match_id` | Unique identifier for each match |
| `character` | Character I played |
| `opponent_character` | Opponent's character |
| `result` | Match outcome: 1 = win, 0 = loss |
| `kos` | Opponent stocks taken |
| `falls` | Stocks lost |
| `sds` | Self-destructs |
| `match_time` | Match duration |
| `damage_given` | Total damage dealt |
| `damage_taken` | Total damage received |
| `gsp` | Global Smash Power |

The dataset currently contains **93 wins and 59 losses**, producing an overall win rate of approximately **61.2%**.

---

# Data Cleaning and Validation

Before performing any analysis, the raw dataset is passed through a cleaning and validation pipeline.

The cleaning process includes:

- Standardizing column names
- Normalizing character names
- Removing whitespace inconsistencies
- Detecting duplicate records
- Checking duplicate match IDs
- Detecting missing values
- Validating numerical columns
- Checking for negative or impossible values
- Converting match duration into seconds
- Validating match outcomes against KOs and falls

A match recorded as a win must satisfy:

```text
KOs > Falls
```

while a recorded loss must satisfy:

```text
KOs < Falls
```

These checks help prevent incorrectly recorded matches from affecting downstream analysis.

The cleaned dataset is saved separately so that all later stages operate on the same validated data.

---

# Exploratory Data Analysis

Exploratory Data Analysis was performed before applying machine learning in order to understand the structure of the dataset and identify important gameplay patterns.

The EDA includes:

- Overall win rate
- Win rate by character
- Average GSP by character
- Damage given vs. damage taken
- KOs by result
- Falls by result
- Match duration by character
- Matchup win-rate heatmap
- Correlation analysis

For matchup analysis, only character-opponent combinations played **at least twice** are included to reduce the influence of one-off encounters.

---

## Character Performance

The strongest character in the current dataset by win rate is **Toon Link**.

| Character | Matches | Wins | Losses | Win Rate |
|---|---:|---:|---:|---:|
| **Toon Link** | 25 | 18 | 7 | **72.0%** |
| Lucas | 26 | 17 | 9 | 65.4% |
| Banjo & Kazooie | 25 | 16 | 9 | 64.0% |
| Captain Falcon | 26 | 15 | 11 | 57.7% |
| Pit | 25 | 14 | 11 | 56.0% |
| Mario | 25 | 13 | 12 | 52.0% |

An interesting result appears when comparing win rate with damage output.

Mario produced the highest average damage among the analyzed characters, yet had the lowest win rate.

Meanwhile, Toon Link achieved the highest win rate without producing the highest raw damage.

This suggests an important theme that appears throughout the project:

> **Dealing more total damage does not necessarily translate into winning more matches. Damage efficiency and the ability to convert advantages into stocks appear to matter more than raw damage output alone.**

---

# Feature Engineering

The original match statistics were transformed into additional features designed to better represent gameplay performance.

## Damage Differential

```text
Damage Differential = Damage Given - Damage Taken
```

A positive value indicates that I dealt more damage than I received.

---

## Damage Ratio

```text
Damage Ratio = Damage Given / Damage Taken
```

This measures damage efficiency.

A ratio greater than `1.0` means I dealt more damage than I received.

---

## KO Differential

```text
KO Differential = KOs - Falls
```

This represents stock advantage during the match.

---

## Damage Per Minute

```text
Damage Per Minute = Damage Given / Match Duration
```

This measures offensive output relative to match pace.

---

## Damage Taken Per Minute

This measures the rate at which the opponent accumulates damage against me.

---

## GSP Change

GSP change measures the difference between the current GSP and the previous match played with the **same character**.

This allows GSP movement to be tracked independently for each character.

---

## Match Intensity

Match intensity measures how frequently KOs and falls occur relative to match duration.

Higher values represent matches with faster stock exchanges and greater overall volatility.

---

# Unsupervised Learning

Two separate K-Means clustering analyses were performed.

The models intentionally answer different questions:

### Match-Level Clustering

> **How do my individual matches tend to play out?**

### Matchup-Level Clustering

> **What types of character matchups do I dominate or struggle against?**

Before clustering, all selected features are standardized using `StandardScaler` because K-Means is distance-based and would otherwise be disproportionately influenced by features with larger numerical scales.

---

# Match-Level Clustering

Individual matches were clustered using gameplay characteristics including:

- KOs
- Falls
- Damage given
- Damage taken
- Match duration
- Damage per minute
- Damage taken per minute
- Match intensity

Importantly, **match result was not used to construct the clusters**.

Win rate was calculated only after clustering.

This allows the analysis to determine whether naturally occurring gameplay patterns are associated with winning without explicitly telling K-Means which matches were wins or losses.

---

## Selecting the Number of Match Clusters

Candidate values of `k` were evaluated using:

- Elbow Method
- Silhouette Score

The highest match-level silhouette score was approximately:

```text
k = 2
Silhouette Score ≈ 0.320
```

Therefore, the final match-level model uses:

```text
K = 2
```

The analysis suggests that my matches naturally separate into **two broad gameplay profiles**.

---

# Match Cluster 0 — Controlled / Extended Matches

Cluster 0 represents longer, more controlled matches.

Average characteristics include:

| Statistic | Cluster 0 |
|---|---:|
| KOs | 2.5 |
| Falls | 2.0 |
| Damage Given | 299.3 |
| Damage Taken | 304.0 |
| Match Duration | 272.7 sec |
| Approx. Match Duration | **4:33** |
| Damage / Minute | 65.8 |
| Damage Taken / Minute | 65.8 |
| Win Rate | **~64%** |

These matches involve relatively balanced total damage and considerably slower damage accumulation.

Despite dealing and receiving almost identical amounts of damage, I average more KOs than falls in this cluster.

Most importantly, the win rate is approximately **64%**, slightly above my overall win rate of approximately 61%.

I interpret this cluster as:

> **Controlled / Extended Matches — longer games characterized by slower damage accumulation, more deliberate exchanges, and slightly above-average winning performance.**

---

# Match Cluster 1 — High-Pressure / Volatile Matches

Cluster 1 represents much faster and more volatile games.

| Statistic | Cluster 1 |
|---|---:|
| KOs | 2.3 |
| Falls | 2.3 |
| Damage Given | 294.2 |
| Damage Taken | 320.0 |
| Match Duration | 164.3 sec |
| Approx. Match Duration | **2:44** |
| Damage / Minute | 114.8 |
| Damage Taken / Minute | 118.4 |
| Win Rate | **~51%** |

Interestingly, total damage is not dramatically different from Cluster 0.

The major difference is **how quickly that damage occurs**.

Damage output increases from approximately:

```text
66 → 115 damage/minute
```

while damage received increases from approximately:

```text
66 → 118 damage/minute
```

These matches therefore involve significantly faster exchanges and much greater volatility.

However, the win rate falls to approximately **51%**.

I interpret this cluster as:

> **High-Pressure / Volatile Matches — shorter games characterized by rapid damage accumulation, frequent exchanges, and below-average winning performance.**

---

# Match-Level Interpretation

The most important finding from the match-level clustering is:

> **I perform better in slower, more controlled matches than in fast, highly volatile games.**

Controlled matches produced approximately a **64% win rate**, while high-pressure matches produced approximately a **51% win rate**.

Because match result was not included in the K-Means feature set, this relationship was discovered **after** clustering rather than being built directly into the clusters.

This suggests that match pace and damage exchange patterns contain meaningful information about my performance.

Character composition also varies between the clusters.

Captain Falcon, Lucas, Mario, and Toon Link appear primarily in the controlled cluster, while Banjo & Kazooie and Pit have substantially larger shares of high-pressure matches.

This does not prove that character choice causes a particular match style, but it suggests that **character selection may be associated with how matches develop statistically**.

---

# Match-Level PCA

Principal Component Analysis was used to project the multidimensional match clusters into two dimensions.

The first two principal components explain:

```text
PC1 = 37.7%
PC2 = 31.1%
```

Together:

```text
Total variance retained ≈ 68.8%
```

The visualization shows noticeable, although not complete, separation between the two match profiles.

This is expected because gameplay styles exist on a continuum rather than as perfectly isolated categories.

---

# Matchup-Level Clustering

The second clustering analysis changes the unit of analysis from individual matches to **character-opponent matchup profiles**.

Matches are grouped by:

```text
My Character + Opponent Character
```

Only combinations played at least twice are retained.

Each matchup is represented using:

- Win rate
- Average damage differential
- Average damage ratio
- Average KO differential
- Average damage per minute
- Average damage taken per minute
- Average match intensity

---

# Selecting the Number of Matchup Clusters

The Elbow Method and Silhouette Score were again used to evaluate candidate cluster counts.

The strongest result occurred at:

```text
k = 4
Silhouette Score ≈ 0.351
```

The inertia curve also showed diminishing improvements around four clusters.

Therefore:

```text
K = 4
```

was selected for the final matchup model.

The four numerical K-Means labels were then interpreted according to their actual statistical profiles rather than assuming that cluster numbers represented rankings.

---

# Matchup Cluster 0 — Severely Difficult

Cluster 0 represents the most difficult matchups in the dataset.

| Statistic | Value |
|---|---:|
| Win Rate | **0%** |
| Avg. Damage Differential | **−150.33** |
| Avg. Damage Ratio | 0.63 |
| Avg. KO Differential | **−1.50** |
| Avg. Damage / Minute | 72.57 |
| Avg. Damage Taken / Minute | 119.46 |

These matchups are not simply losses.

The underlying statistics indicate a substantial performance disadvantage.

On average, I receive approximately **150 more damage than I deal**, and my damage ratio of `0.63` indicates that I deal only about 63% as much damage as I receive.

The −1.50 KO differential also indicates a large stock disadvantage.

Examples include:

- Banjo & Kazooie vs. Wii Fit Trainer
- Pit vs. Steve
- Mario vs. Captain Falcon

I interpret this cluster as:

> **Severely Difficult Matchups — matchups characterized by large damage deficits, negative stock differentials, and consistently poor outcomes.**

---

# Matchup Cluster 1 — Dominant

Cluster 1 represents my strongest matchup performances.

| Statistic | Value |
|---|---:|
| Win Rate | **79%** |
| Avg. Damage Differential | **+61.44** |
| Avg. Damage Ratio | **1.42** |
| Avg. KO Differential | **+1.10** |
| Avg. Damage / Minute | 71.00 |
| Avg. Damage Taken / Minute | 54.65 |

These matchups show clear statistical advantages.

A damage ratio of `1.42` means that I deal approximately **42% more damage than I receive** on average.

The +1.10 KO differential also indicates that these advantages translate into stocks rather than simply accumulating damage.

Examples include:

- Mario vs. Mii Brawler
- Toon Link vs. Ike
- Toon Link vs. Lucina
- Lucas vs. Ganondorf

I interpret this cluster as:

> **Dominant Matchups — favorable matchups characterized by high win rates, strong damage efficiency, and substantial stock advantages.**

---

# Matchup Cluster 2 — High-Intensity Favorable

Cluster 2 represents an especially interesting category.

| Statistic | Value |
|---|---:|
| Win Rate | **63%** |
| Avg. Damage Differential | +7.37 |
| Avg. Damage Ratio | 1.10 |
| Avg. KO Differential | +0.63 |
| Avg. Damage / Minute | **108.04** |
| Avg. Damage Taken / Minute | **107.21** |
| Match Intensity | **Highest of the four clusters** |

Unlike the dominant cluster, these matches do not contain a large damage advantage.

Instead, both players accumulate damage extremely quickly.

Despite nearly equal damage exchange, I maintain a positive KO differential and win approximately **63%** of these matchups.

I interpret this cluster as:

> **High-Intensity Favorable Matchups — fast, volatile matchups with nearly equal damage exchange but a positive stock advantage and above-average win rate.**

---

# Matchup Cluster 3 — Competitive / Slightly Unfavorable

Cluster 3 represents relatively balanced matchups that currently lean slightly toward the opponent.

| Statistic | Value |
|---|---:|
| Win Rate | **48%** |
| Avg. Damage Differential | −31.56 |
| Avg. Damage Ratio | 0.92 |
| Avg. KO Differential | −0.24 |
| Avg. Damage / Minute | 59.92 |
| Avg. Damage Taken / Minute | 67.50 |

Unlike the severely difficult cluster, these matches remain competitive.

The damage ratio is close to `1.0`, the KO differential is only slightly negative, and the win rate is close to 50%.

I interpret this cluster as:

> **Competitive / Slightly Unfavorable Matchups — relatively balanced matchups where the opponent currently maintains a small statistical advantage.**

---

# Matchup-Level PCA

PCA provides an especially strong visualization for the matchup clusters.

The first two components explain:

```text
PC1 = 56.1%
PC2 = 34.1%
```

Together they retain approximately:

```text
90.2% of total variance
```

This means the two-dimensional PCA visualization preserves the vast majority of the variation contained in the original clustering features.

The resulting plot shows meaningful spatial separation between the four matchup profiles.

Severely difficult matchups occupy a very different region from dominant matchups, while competitive and high-intensity matchups form their own patterns.

This provides visual support that the K-Means clusters correspond to meaningfully different statistical profiles rather than arbitrary numerical assignments.

---

# Supervised Learning — XGBoost

The final stage treats match performance as a supervised classification problem.

An **XGBoost classifier** is trained to predict whether a match results in a win or loss.

Several variables were deliberately excluded to reduce outcome leakage and prevent the model from simply reconstructing the final scoreboard.

Excluded variables include:

- KOs
- Falls
- KO differential
- GSP change

Self-destructs are also excluded when they contain no meaningful variation.

Instead, the model uses gameplay variables such as:

- Damage given
- Damage taken
- Damage differential
- Damage ratio
- Damage per minute
- Damage taken per minute
- Match intensity
- GSP
- Match duration
- Character
- Opponent familiarity

This makes the supervised analysis more meaningful because the model must identify relationships between gameplay characteristics and outcomes rather than relying directly on the stock count that determines the winner.

---

# XGBoost Evaluation

The model was evaluated using a stratified holdout test set.

The confusion matrix was:

| | Predicted Loss | Predicted Win |
|---|---:|---:|
| **Actual Loss** | **8** | 4 |
| **Actual Win** | 6 | **13** |

The model correctly classified:

```text
21 / 31 matches
```

producing a holdout accuracy of approximately:

```text
67.7%
```

The model correctly identified:

```text
13 / 19 wins ≈ 68%
8 / 12 losses ≈ 67%
```

This indicates relatively balanced performance between the two outcome classes rather than simply predicting the majority class.

---

# ROC-AUC

The XGBoost model achieved:

```text
ROC-AUC = 0.84
```

An ROC-AUC of `0.50` would indicate random discrimination.

The observed value of **0.84** indicates that the model has good ability to distinguish winning matches from losing matches across classification thresholds.

Given the relatively small size and personal nature of the dataset, this result suggests that the recorded gameplay statistics contain meaningful predictive information about match outcomes.

---

# Feature Importance

XGBoost gain-based feature importance identified several influential variables.

The strongest feature was:

```text
damage_ratio
```

followed by variables including:

- Character information
- Damage differential
- Damage taken per minute
- Damage per minute
- GSP
- Match intensity
- Raw damage statistics
- Match duration

One of the most important findings is that **relative damage measurements are more informative than raw damage alone**.

Damage ratio and damage differential rank more prominently than simply measuring total damage given.

This reinforces the result observed during EDA:

> **Successful performance appears to depend more on damage efficiency than on maximizing raw damage output.**

---

# SHAP Interpretation

SHAP analysis was used to determine both the magnitude **and direction** of feature contributions.

Unlike traditional feature importance, which primarily indicates how useful a variable is to the model, SHAP helps explain whether particular feature values push predictions toward a **win** or a **loss**.

---

## Damage Ratio

Damage ratio produces the clearest relationship in the SHAP analysis.

High damage ratios generally produce positive SHAP values, pushing predictions toward:

```text
WIN
```

while low damage ratios produce strongly negative SHAP values, pushing predictions toward:

```text
LOSS
```

This provides strong evidence that **damage efficiency is one of the clearest measurable indicators of successful play in the dataset**.

---

## Damage Differential

Damage differential reinforces the same conclusion.

Positive damage differentials generally move model predictions toward wins, while negative differentials tend to move predictions toward losses.

Together:

```text
Higher Damage Ratio
        +
Higher Damage Differential
        ↓
Greater Predicted Probability of Winning
```

These two variables describe performance relative to the opponent rather than simply measuring offensive volume.

---

## Raw Damage

Raw damage given contributes to the model, but it is less influential than relative damage measures.

This distinction is important.

For example:

```text
350 damage dealt
```

does not necessarily represent strong performance if:

```text
450 damage was received
```

The model appears to recognize this difference.

Therefore:

> **Winning is less about dealing the largest possible amount of damage and more about dealing damage efficiently relative to what the opponent is doing.**

---

## GSP

GSP also contributes meaningfully to predictions.

Higher GSP values generally appear more frequently on the positive side of the SHAP output, while lower values more frequently push predictions toward losses.

However, this should be interpreted as an **association**, not a causal relationship.

GSP may capture several underlying factors related to current performance, matchmaking environment, and recent results.

---

## Character Effects

Character choice also provides predictive information.

Captain Falcon receives relatively high gain-based feature importance, while SHAP shows character-specific effects for some observations.

However, character importance should not be interpreted as directly ranking the characters.

For example, high importance for Captain Falcon does **not** mean Captain Falcon is my strongest character.

EDA shows that Toon Link currently has the highest win rate.

Instead, the appropriate interpretation is:

> **Character choice provides additional predictive information after gameplay statistics are considered, but its relationship with winning is more complex than a simple character ranking.**

---

# Overall Interpretation

The different stages of the project converge on a consistent story about my gameplay.

## 1. Character choice matters, but raw damage does not determine the best character.

Toon Link currently produces my highest win rate at **72%**.

Mario produces higher average raw damage but has a substantially lower win rate.

Therefore, simply dealing more damage does not necessarily produce better outcomes.

---

## 2. I perform better in controlled matches.

K-Means discovered two natural match profiles without using match outcome.

Longer, controlled matches produced approximately a:

```text
64% win rate
```

while shorter, high-pressure matches produced approximately:

```text
51% win rate
```

This suggests that my performance improves when matches develop at a slower and more controlled pace.

---

## 3. Matchups fall into distinct performance categories.

Matchup clustering identified four meaningful groups:

```text
Dominant
High-Intensity Favorable
Competitive / Slightly Unfavorable
Severely Difficult
```

These groups differ not only in win rate but also in:

- Damage efficiency
- KO advantage
- Damage pace
- Match intensity

This demonstrates that matchup difficulty cannot be adequately described using win rate alone.

---

## 4. Damage efficiency is the strongest recurring performance signal.

This conclusion appears repeatedly across the project.

EDA shows that high raw damage does not guarantee a high win rate.

Matchup clustering shows dominant matchups have strong positive damage ratios and damage differentials.

XGBoost identifies damage ratio as its strongest feature.

SHAP shows that high damage ratios push predictions strongly toward wins.

Therefore, one of the clearest overall conclusions is:

> **My strongest performances occur when I efficiently out-damage the opponent, rather than simply producing large amounts of total damage.**

---

## 5. Machine learning provides complementary perspectives.

The project does not rely on a single model.

Instead:

```text
EDA
↓
What happened?

K-Means
↓
What natural performance patterns exist?

PCA
↓
How are those patterns structured visually?

XGBoost
↓
Can gameplay statistics predict outcomes?

SHAP
↓
Why does the model make those predictions?
```

Together, these approaches provide a much more complete picture of personal performance than simply recording wins and losses.

---

# Key Findings

The main findings from the current dataset are:

- **Overall Win Rate:** ~61.2%
- **Best Character by Win Rate:** Toon Link — **72%**
- **Match-Level K-Means:** `k = 2`
- **Controlled Match Win Rate:** ~64%
- **High-Pressure Match Win Rate:** ~51%
- **Match-Level PCA Variance Retained:** ~68.8%
- **Matchup-Level K-Means:** `k = 4`
- **Dominant Matchup Cluster Win Rate:** ~79%
- **Severely Difficult Matchup Cluster Win Rate:** 0%
- **Matchup PCA Variance Retained:** ~90.2%
- **XGBoost Holdout Accuracy:** ~67.7%
- **XGBoost ROC-AUC:** **0.84**
- **Strongest Predictive Signal:** Damage Ratio / Damage Efficiency

---

# Limitations

This project is designed as a **self-performance analytics system**, not as a universal model of competitive Super Smash Bros. Ultimate.

The dataset represents a single player's matches.

Therefore, conclusions describe **my own gameplay tendencies** and should not be generalized to all Smash players.

The dataset is also relatively small at 152 matches.

Some individual character matchups contain few observations, which is why matchup-level analysis requires at least two matches before a matchup is included.

The XGBoost model should similarly be interpreted as an exploratory personal prediction model rather than evidence of general competitive predictive performance.

Additional data will allow the stability of these findings to be tested over time.

---

# Future Work

The project is designed so that additional matches can be continuously added to the dataset.

Future improvements include:

- Collecting substantially more matches
- Increasing observations per character
- Increasing observations for repeated matchups
- Tracking performance changes over time
- Comparing cluster stability as the dataset grows
- Examining character-specific gameplay styles
- Evaluating whether matchup categories remain stable
- Retraining XGBoost on larger datasets
- Comparing predictive performance across characters
- Building an interactive performance dashboard
- Automating the analytics pipeline as new matches are collected

The long-term goal is to transform the project into a continuously updated **personal Super Smash Bros. performance analytics system**.

---

# Project Structure

```text
Smash-bros-Self-Performance-Analytics/
│
├── Smash bros games.csv
│
├── data_cleaning.py
├── eda.py
├── feature_engineering.py
├── matchup_aggregation.py
├── clustering1.py
├── clustering2.py
├── xgboost_model.py
│
├── cleaned_smash_dataset.csv
├── engineered_smash_dataset.csv
├── matchup_features.csv
├── clustered_matchups.csv
├── clustered_smash_dataset.csv
│
└── outputs/
    ├── eda/
    ├── matchup_clustering/
    ├── match_clustering/
    └── xgboost/
```

---

# Technologies

### Data Processing
- Python
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- K-Means Clustering
- Principal Component Analysis
- XGBoost

### Model Evaluation & Explainability
- SHAP

---

# Conclusion

This project demonstrates how a relatively small, self-collected dataset can be transformed into a complete **data science and machine learning workflow**.

Rather than simply calculating win rates, the analysis combines descriptive statistics, engineered domain-specific features, unsupervised learning, dimensionality reduction, supervised classification, and model explainability to investigate how I actually perform in Super Smash Bros. Ultimate.

The analysis identified **Toon Link as my strongest character by win rate**, discovered that I achieve better results in **slower and more controlled matches**, separated character matchups into **four distinct performance profiles**, and showed through XGBoost and SHAP that **damage efficiency is one of the strongest measurable indicators of winning**.

Most importantly, the project demonstrates that performance is multidimensional.

Winning is not simply a function of dealing the most damage. It is associated with **how efficiently damage is exchanged, how quickly the match develops, which character and matchup are involved, and how effectively advantages are converted into successful outcomes**.

As additional matches are collected, the same pipeline can be rerun to determine whether these patterns remain stable or evolve alongside my gameplay.
