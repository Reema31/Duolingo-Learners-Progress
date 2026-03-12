import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
)
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Duolingo Learners Progress",
    page_icon="🦜",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:800; color:#58cc02;
                   text-align:center; padding-bottom:.2rem; }
    .sub-title   { text-align:center; color:#666; font-size:1rem;
                   margin-bottom:2rem; }
    .sec-header  { font-size:1.35rem; font-weight:700; color:#1cb0f6;
                   border-left:5px solid #58cc02; padding-left:.6rem;
                   margin-top:2rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🦜 Duolingo Learners Progress</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Memory Effect &amp; Language Learning Analytics</div>',
    unsafe_allow_html=True,
)

# ── Load data — bundled CSV first, optional upload as override ────────────────
BUNDLED = os.path.join(os.path.dirname(__file__), "duolingoSample.csv")

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/"
        "Duolingo_logo.svg/1280px-Duolingo_logo.svg.png",
        width=160,
    )
    st.markdown("### 📂 Data")
    uploaded = st.file_uploader(
        "Upload your own CSV (optional)",
        type=["csv"],
        help="Must have the same columns as learning_traces.csv",
    )
    st.markdown("---")
    st.markdown(
        "**Sections**\n"
        "- 📊 Data Overview\n"
        "- 👤 User Behavior\n"
        "- 📈 User Performance\n"
        "- 🧠 Memory Analysis\n"
        "- 🤖 Models",
    )


@st.cache_data(show_spinner="Loading data…")
def load(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


raw = load(uploaded if uploaded else BUNDLED)

# ── Cleaning (mirrors the notebook) ──────────────────────────────────────────
@st.cache_data(show_spinner="Preparing data…")
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df.rename(columns={
        "p_recall":      "recall_accuracy",
        "timestamp":     "lesson_time",
        "delta":         "time_since_last_seen",
        "lexeme_id":     "word_id",
        "lexeme_string": "word",
    }, inplace=True)

    df.drop_duplicates(inplace=True)
    df["days_since_last_seen"] = df["time_since_last_seen"] // 86_400
    df["lesson_time"] = pd.to_datetime(
        df["lesson_time"].astype("int64"), unit="s", errors="coerce"
    )
    df["hour"] = df["lesson_time"].dt.hour
    df["date"] = df["lesson_time"].dt.date

    def clock(h):
        if   4 <= h <= 11: return "Early Bird"
        elif 12 <= h <= 20: return "Pigen"
        else:               return "Night Owl"

    df["learner_type"] = df["hour"].apply(clock)

    ratio = df["history_correct"] / df["history_seen"].replace(0, np.nan)
    df["learning_speed"] = ratio.apply(
        lambda x: "Fast" if x >= 0.9 else ("Average" if x >= 0.6 else "Slow")
        if pd.notna(x) else "Slow"
    )

    df["session_accuracy"] = (
        df["session_correct"] / df["session_seen"].replace(0, np.nan)
    )
    df["history_accuracy"] = (
        df["history_correct"] / df["history_seen"].replace(0, np.nan)
    )
    return df


df = clean(raw)

# helper so every plot renders the same way
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs(
    ["📊 Data Overview", "👤 User Behavior",
     "📈 User Performance", "🧠 Memory Analysis", "🤖 Models"]
)

# ── TAB 1 · Data Overview ─────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="sec-header">Dataset at a Glance</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",       f"{len(df):,}")
    c2.metric("Unique Users",         f"{df['user_id'].nunique():,}")
    c3.metric("Languages Covered",    f"{df['learning_language'].nunique()}")
    c4.metric("Avg Recall Accuracy",  f"{df['recall_accuracy'].mean():.3f}")

    st.markdown("#### Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)
    with col2:
        st.markdown("#### Missing Values")
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        st.dataframe(miss, use_container_width=True)

# ── TAB 2 · User Behavior ─────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="sec-header">Language Distribution</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Learning Language (unique users)**")
        uniq_lan = df.groupby("learning_language")["user_id"].nunique()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(uniq_lan.values, labels=uniq_lan.index,
               colors=sns.color_palette("Set2", len(uniq_lan)),
               autopct="%1.1f%%", startangle=140)
        ax.set_title("Learning Language Distribution")
        show(fig)

    with col2:
        st.markdown("**UI Language (unique users)**")
        uniq_ui = df.groupby("ui_language")["user_id"].nunique()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(uniq_ui.values, labels=uniq_ui.index,
               colors=sns.color_palette("Set3", len(uniq_ui)),
               autopct="%1.1f%%", startangle=140)
        ax.set_title("UI Language Distribution")
        show(fig)

    st.markdown('<div class="sec-header">Learner Type</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="learner_type", palette="colorblind", ax=ax)
        ax.set_title("Learner Type Distribution")
        ax.set_xlabel("Learner Type"); ax.set_ylabel("Count")
        plt.tight_layout(); show(fig)

    with col2:
        st.markdown(
            "Derived from **hour of study**:\n\n"
            "- 🌅 **Early Bird**: 4 AM – 11 AM\n"
            "- ☀️ **Pigen**: 12 PM – 8 PM\n"
            "- 🌙 **Night Owl**: 9 PM – 3 AM"
        )
        st.dataframe(
            df["learner_type"].value_counts().rename_axis("Type")
              .reset_index(name="Count"),
            use_container_width=True,
        )

    st.markdown('<div class="sec-header">Learning Speed</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="learning_speed", palette="Set3",
                      order=["Fast", "Average", "Slow"], ax=ax)
        ax.set_title("Distribution of Learning Speed")
        ax.set_xlabel("Learning Speed"); ax.set_ylabel("Count")
        plt.tight_layout(); show(fig)

    with col2:
        st.markdown(
            "Based on **history_correct / history_seen**:\n\n"
            "- 🚀 **Fast**: ≥ 90 %\n"
            "- 🟡 **Average**: 60–89 %\n"
            "- 🐢 **Slow**: < 60 %"
        )
        st.dataframe(
            df["learning_speed"].value_counts().rename_axis("Speed")
              .reset_index(name="Count"),
            use_container_width=True,
        )

# ── TAB 3 · User Performance ──────────────────────────────────────────────────
with t3:
    st.markdown('<div class="sec-header">Daily Platform Activity</div>',
                unsafe_allow_html=True)

    daily = (
        df.groupby("date")
          .agg(active_users=("user_id", "nunique"),
               correct_answers=("history_correct", "sum"))
          .reset_index().sort_values("date")
    )

    fig, ax1 = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=daily, x="date", y="active_users",
                 ax=ax1, color="steelblue", label="Active Users")
    ax1.set_ylabel("Active Users", color="steelblue")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    sns.lineplot(data=daily, x="date", y="correct_answers",
                 ax=ax2, color="green", label="Correct Answers")
    ax2.set_ylabel("Correct Answers", color="green")
    ax1.set_title("Daily Platform Activity")
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0]+lines2[0], lines[1]+lines2[1], loc="upper left")
    ax2.get_legend().remove()
    plt.tight_layout(); show(fig)

    st.markdown('<div class="sec-header">Session vs History Accuracy by Language</div>',
                unsafe_allow_html=True)

    lang_acc = (
        df.groupby("learning_language")
          .agg(avg_session_accuracy=("session_accuracy", "mean"),
               avg_history_accuracy=("history_accuracy", "mean"))
          .reset_index()
    )
    lang_long = lang_acc.melt(
        id_vars="learning_language",
        value_vars=["avg_session_accuracy", "avg_history_accuracy"],
        var_name="Accuracy_Type", value_name="Accuracy",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=lang_long, x="learning_language", y="Accuracy",
                hue="Accuracy_Type", palette="deep", ax=ax)
    ax.set_title("Session vs History Accuracy by Learning Language")
    ax.set_xlabel("Learning Language")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); show(fig)

    st.markdown('<div class="sec-header">Estimated Language Difficulty</div>',
                unsafe_allow_html=True)

    lang_diff = df.groupby("learning_language")["session_correct"].mean().reset_index()
    lang_diff["difficulty"] = lang_diff["session_correct"] - 1

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.pointplot(data=lang_diff, x="learning_language", y="difficulty",
                  color="purple", ax=ax)
    ax.set_title("Estimated Difficulty by Learning Language")
    ax.set_xlabel("Learning Language"); ax.set_ylabel("Difficulty Score")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); show(fig)

    st.markdown('<div class="sec-header">Cross-Linguistic Transfer Heatmap</div>',
                unsafe_allow_html=True)
    st.caption("Recall rate by UI (native) → Learning language. Groups with < 5 rows excluded.")

    pivot = (
        df.assign(correct_binary=df["session_correct"].gt(0).astype(int))
          .groupby(["ui_language", "learning_language"])
          .filter(lambda x: len(x) >= 5)
          .groupby(["ui_language", "learning_language"])
          .correct_binary.mean()
          .unstack()
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma",
                    cbar_kws={"label": "Recall Rate"}, ax=ax)
        ax.set_title("Cross-Linguistic Transfer: Recall Rate by UI → Learning Language")
        ax.set_xlabel("Learning Language"); ax.set_ylabel("UI (Native) Language")
        plt.tight_layout(); show(fig)
    else:
        st.info("Not enough language-pair data for the heatmap.")

# ── TAB 4 · Memory Analysis ───────────────────────────────────────────────────
with t4:
    st.markdown('<div class="sec-header">Forgetting Curve</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Memory decays sharply right after learning, then levels off. "
        "Duolingo's **spaced repetition** schedules reviews at the optimal moment "
        "to fight this curve and build long-term retention."
    )

    df_fc = df.copy()
    # use fewer bins for a small dataset so we don't get empty buckets
    n_bins = min(100, max(10, len(df_fc) // 10))
    df_fc["time_bin"] = pd.cut(
        df_fc["days_since_last_seen"],
        bins=np.logspace(-2, 3, n_bins),
    )
    curve = (
        df_fc.groupby("time_bin", observed=True)
             .agg(recall_rate=("recall_accuracy", "mean"),
                  time=("days_since_last_seen", "median"))
             .dropna()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve["time"], curve["recall_rate"],
            marker="o", color="green", markersize=4, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Time since last review (days) — log scale")
    ax.set_ylabel("Recall Probability")
    ax.set_title("Forgetting Curve")
    ax.grid(True, alpha=0.4)
    plt.tight_layout(); show(fig)

    st.markdown('<div class="sec-header">Correlation Heatmap</div>',
                unsafe_allow_html=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    plt.tight_layout(); show(fig)

# ── TAB 5 · Models ────────────────────────────────────────────────────────────
with t5:
    # ── Regression ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-header">Model 1 — Linear & Random Forest Regression</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Predicting **recall accuracy** from time gap, language, and history stats.")

    @st.cache_data(show_spinner="Training regression models…")
    def train_regression(df: pd.DataFrame):
        feats = ["time_since_last_seen", "learning_language", "ui_language",
                 "history_seen", "history_correct"]
        X = pd.get_dummies(df[feats], columns=["learning_language", "ui_language"],
                           drop_first=True)
        y = df["recall_accuracy"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression
        lr = LinearRegression().fit(X_tr, y_tr)
        yp_lr  = lr.predict(X_te)
        rmse_lr = np.sqrt(mean_squared_error(y_te, yp_lr))
        r2_lr   = r2_score(y_te, yp_lr)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        yp_rf  = rf.predict(X_te)
        mse_rf = mean_squared_error(y_te, yp_rf)
        r2_rf  = r2_score(y_te, yp_rf)

        return y_te, yp_lr, rmse_lr, r2_lr, yp_rf, mse_rf, r2_rf, rf, X_tr.columns.tolist()

    y_te, yp_lr, rmse_lr, r2_lr, yp_rf, mse_rf, r2_rf, rf_m, feat_cols = train_regression(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linear RMSE", f"{rmse_lr:.4f}")
    c2.metric("Linear R²",   f"{r2_lr:.4f}")
    c3.metric("RF MSE",       f"{mse_rf:.4f}")
    c4.metric("RF R²",        f"{r2_rf:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Actual vs Predicted (Linear Regression)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_te, yp_lr, alpha=0.35, color="steelblue", s=12)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.5)
        ax.set_xlabel("Actual Recall Accuracy")
        ax.set_ylabel("Predicted")
        ax.set_title("Linear Regression: Actual vs Predicted")
        plt.tight_layout(); show(fig)

    with col2:
        st.markdown("**Feature Importance (Random Forest — top 12)**")
        imp = pd.Series(rf_m.feature_importances_, index=feat_cols).nlargest(12)
        fig, ax = plt.subplots(figsize=(5, 4))
        imp.sort_values().plot(kind="barh", color="seagreen", ax=ax)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        plt.tight_layout(); show(fig)

    # ── Classification ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-header">Model 2 — Logistic Regression (Learner Type Classifier)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Classifying **learner type** from session accuracy and learning speed.")

    @st.cache_data(show_spinner="Training logistic regression…")
    def train_logistic(df: pd.DataFrame):
        X = pd.get_dummies(
            df[["session_accuracy", "learning_speed"]].fillna(0),
            drop_first=True,
        )
        y_cat  = df["learner_type"].astype("category")
        cats   = y_cat.cat.categories
        y_code = y_cat.cat.codes
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_code, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        yp  = clf.predict(X_te)
        acc = accuracy_score(y_te, yp)
        rep = classification_report(y_te, yp, target_names=cats, output_dict=True)
        cm  = confusion_matrix(y_te, yp)
        return acc, rep, cats, cm

    acc, rep, cats, cm = train_logistic(df)

    st.metric("Logistic Regression Accuracy", f"{acc:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)
    with col2:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cats, yticklabels=cats, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout(); show(fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:.85rem'>"
    "Duolingo Spaced Repetition Analysis · Built with Streamlit 🦜"
    "</div>",
    unsafe_allow_html=True,
)
