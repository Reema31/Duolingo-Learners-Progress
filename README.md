# Duolingo Learners Progress & Memory Effect in Language Learning

Analyzing how Duolingo learners acquire and retain vocabulary over time, with a focus on the memory effect and spaced repetition patterns.


📌 Project Overview
Duolingo is one of the most widely used platforms for language learning, offering millions of learners interactive and personalized practice. This project explores learner behavior and memory retention patterns by analyzing spaced repetition data — examining how repeated exposure and timed review sessions improve long-term vocabulary recall.
Key questions explored:

Which languages are most commonly learned on Duolingo?
How does time since last review affect recall accuracy (Forgetting Curve)?
What cross-linguistic patterns exist between a user's native language and the language they are learning?
Can we predict recall accuracy using machine learning?


⚠️ Dataset Note

The original dataset was too large to implement on github and streamlit, so a sample dataset was used for this analysis to keep computation manageable.

The original full dataset (~12.8 million learning events) is publicly available on Kaggle:
🔗 Duolingo Spaced Repetition Data — Kaggle
The dataset contains learning traces from real Duolingo users, including features such as recall probability, time since last review, session correctness, and learner history.

Project Structure
duolingo_learners_progress.py   # Main analysis notebook/script
README.md                       # Project documentation

Libraries & Tools
Library      Purpose
pandas       Data loading, cleaning, and manipulation
numpy        Numerical operations
matplotlib   Data visualization
seaborn      Statistical plotting
scikit-learn Machine learning models and evaluation

Analysis Sections
1. Data Loading & Structure
2. 
Loaded the dataset and explored shape, data types, and statistical summaries
Identified that the data contains 12.8M+ learning events across multiple languages

3. Data Cleaning

Renamed columns for readability (e.g. p_recall → recall_accuracy, delta → time_since_last_seen)
Removed duplicate rows
Converted timestamps to datetime format
Derived days_since_last_seen from raw seconds

3. User Behavior Analysis

Learning Language Distribution — Pie chart showing which languages users are studying
Session Performance — Correct vs. incorrect answer trends
Cross-Linguistic Transfer Heatmap — Recall rates broken down by UI language and learning language combinations

4. Memory & Forgetting Curve

Plotted the classic Forgetting Curve — showing how recall probability drops as time since last review increases
Generated a Correlation Heatmap across all numerical features

5. Machine Learning Models
Model 1: Linear Regression & Random Forest Regressor

Target: recall_accuracy (probability of recalling a word)
Features: Time since last seen, learning language, UI language, history seen, history correct
Categorical variables encoded with one-hot encoding
Evaluated using RMSE, MSE, and R² Score

Model 2: Logistic Regression

Target: learner_type (categorical classification)
Features: session_accuracy, learning_speed
Evaluated using Accuracy Score and a full Classification Report


Key Findings

The Forgetting Curve confirms that recall drops sharply in the first few days after a review, validating Duolingo's spaced repetition approach
Cross-linguistic transfer patterns are visible — some native language / target language pairs show notably higher recall rates
Random Forest outperforms Linear Regression for predicting recall accuracy
Logistic Regression successfully classifies learner types based on session behavior



References

Dataset: Duolingo Spaced Repetition Data on Kaggle
Ebbinghaus Forgetting Curve — foundational memory research behind spaced repetition systems


📝 License
This project is for educational and research purposes only. The dataset is provided by Kaggle and subject to its original licensing terms.
