# Duolingo Learners Progress & Memory Effect in Language Learning

> Analyzing how Duolingo learners acquire and retain vocabulary over time, with a focus on the **memory effect** and spaced repetition patterns.

---
![DuoHi](https://github.com/user-attachments/assets/4ca4ad05-2ce5-4b27-b05d-744bcf5b2df9)

## Project Overview

Duolingo is one of the most widely used platforms for language learning, offering millions of learners interactive and personalized practice. This project explores learner behavior and memory retention patterns by analyzing spaced repetition data — examining how repeated exposure and timed review sessions improve long-term vocabulary recall.

Key questions explored:
- Which languages are most commonly learned on Duolingo?
- How does time since last review affect recall accuracy (Forgetting Curve)?
- What cross-linguistic patterns exist between a user's native language and the language they are learning?
- Can we predict recall accuracy using machine learning?

---

## Dataset Note

> **The original dataset was too large to implement in github and streamlit directly**, so a **sample dataset** was used for this analysis to keep computation manageable.

The original full dataset (~12.8 million learning events) is publicly available on Kaggle:

🔗 [Duolingo Spaced Repetition Data — Kaggle](https://www.kaggle.com/datasets/aravinii/duolingo-spaced-repetition-data)

The dataset contains learning traces from real Duolingo users, including features such as recall probability, time since last review, session correctness, and learner history.

---

## Project Structure

```
duolingo_learners_progress.py   # Main analysis notebook/script
README.md                       # Project documentation
```

---

## Libraries & Tools

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plotting |
| `scikit-learn` | Machine learning models and evaluation |

---

## Analysis Sections

### 1. Data Loading & Structure
- Loaded the dataset and explored shape, data types, and statistical summaries
- Identified that the data contains 12.8M+ learning events across multiple languages

### 2. Data Cleaning
- Renamed columns for readability (e.g. `p_recall` → `recall_accuracy`, `delta` → `time_since_last_seen`)
- Removed duplicate rows
- Converted timestamps to datetime format
- Derived `days_since_last_seen` from raw seconds

### 3. User Behavior Analysis
- **Learning Language Distribution** — Pie chart showing which languages users are studying
- **Session Performance** — Correct vs. incorrect answer trends
- **Cross-Linguistic Transfer Heatmap** — Recall rates broken down by UI language and learning language combinations

### 4. Memory & Forgetting Curve
- Plotted the classic **Forgetting Curve** — showing how recall probability drops as time since last review increases
- Generated a **Correlation Heatmap** across all numerical features

### 5. Machine Learning Models

#### Model 1: Linear Regression & Random Forest Regressor
- **Target:** `recall_accuracy` (probability of recalling a word)
- **Features:** Time since last seen, learning language, UI language, history seen, history correct
- Categorical variables encoded with one-hot encoding
- Evaluated using **RMSE**, **MSE**, and **R² Score**

#### Model 2: Logistic Regression
- **Target:** `learner_type` (categorical classification)
- **Features:** `session_accuracy`, `learning_speed`
- Evaluated using **Accuracy Score** and a full **Classification Report**

---

## Key Findings

- The **Forgetting Curve** confirms that recall drops sharply in the first few days after a review, validating Duolingo's spaced repetition approach
- Cross-linguistic transfer patterns are visible — some native language / target language pairs show notably higher recall rates
- Random Forest outperforms Linear Regression for predicting recall accuracy
- Logistic Regression successfully classifies learner types based on session behavior

---

## Key Differences between working on the original dataset (in duolingo_learners_progress) and sample dataset (in app.py):
Due to the reduced sample size, some visualizations differ from the full dataset results. Notable differences include:

The forgetting Curve

<img width="899" height="694" alt="Screenshot 2026-03-12 144847" src="https://github.com/user-attachments/assets/2343d27b-7c4e-45c0-b340-4b60a9eaa469" />

Estimated Difficulty by Learning Language

<img width="875" height="546" alt="Screenshot 2026-03-12 145055" src="https://github.com/user-attachments/assets/2e84a0c5-0838-4e23-9aad-308c055bc255" />

## References

- Dataset: [Duolingo Spaced Repetition Data on Kaggle](https://www.kaggle.com/datasets/aravinii/duolingo-spaced-repetition-data)
- Ebbinghaus Forgetting Curve — foundational memory research behind spaced repetition systems

---

## License

This project is for educational and research purposes only. The dataset is provided by Kaggle and subject to its original licensing terms.
