"""Beginner-friendly Streamlit dashboard for the fake-news demo classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f7f9fc 0%, #edf3fa 100%); }
    .hero {
        padding: 1.8rem 2rem; border-radius: 18px; color: white;
        background: linear-gradient(110deg, #142d4e, #2563a6);
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin-bottom: .35rem; }
    .hero p { color: #e7effa; font-size: 1.05rem; margin-bottom: 0; }
    .notice {
        border-left: 5px solid #e7a12d; background: #fff7e6;
        padding: .8rem 1rem; border-radius: 8px; margin: .8rem 0 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load and validate the CSV used by the app's charts and examples."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset file is missing: {DATASET_PATH}")
    data = pd.read_csv(DATASET_PATH)
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("dataset.csv must have 'text' and 'label' columns.")
    data = data.loc[:, ["text", "label"]].dropna().copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.upper()
    data = data[data["text"].ne("") & data["label"].isin(["REAL", "FAKE"])]
    if data.empty:
        raise ValueError("The CSV contains no usable REAL or FAKE news records.")
    return data.reset_index(drop=True)


@st.cache_resource
def load_model_files():
    """Load trained artifacts; show actionable errors instead of guessing."""
    missing = [path.name for path in (MODEL_PATH, VECTORIZER_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing trained file(s): {', '.join(missing)}. "
            "Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def show_hero() -> None:
    st.markdown(
        '<div class="hero"><h1>📰 Fake News Detection System</h1>'
        "<p>A beginner-friendly Data Mining mini project using Python, "
        "NLP, TF-IDF and Logistic Regression.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice"><b>Important:</b> Predictions are automated '
        "estimates from a small demonstration dataset. They are not verified "
        "facts and should not be used as the sole basis for sharing or "
        "judging a news story.</div>",
        unsafe_allow_html=True,
    )


st.sidebar.title("🧭 Project Menu")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔎 Fake News Prediction", "📊 Dataset Analysis", "🎯 Model Performance"],
)

try:
    dataset = load_dataset()
    model, vectorizer = load_model_files()
    metrics = (
        json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if METRICS_PATH.is_file()
        else None
    )
except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
    st.error(f"Could not load the project data: {error}")
    st.stop()

show_hero()

if page == "🏠 Home":
    st.subheader("Welcome")
    st.write(
        "Explore how a simple text-classification workflow can turn news "
        "articles into numerical features and make a REAL/FAKE prediction. "
        "Use the menu to try a prediction, inspect the CSV, or review the "
        "model's held-out evaluation."
    )
    counts = dataset["label"].value_counts()
    real_count = int(counts.get("REAL", 0))
    fake_count = int(counts.get("FAKE", 0))
    first, second, third = st.columns(3)
    first.metric("🗂️ News records", f"{len(dataset):,}")
    second.metric("✅ Real records", f"{real_count:,}")
    third.metric("⚠️ Fake records", f"{fake_count:,}")
    st.info(
        "Workflow: clean text → remove stopwords → TF-IDF → Logistic "
        "Regression → evaluate on a held-out test split."
    )

elif page == "🔎 Fake News Prediction":
    st.subheader("Try a news prediction")
    st.write("Paste a headline or article below. Longer text may provide more context.")
    samples = dataset.sample(n=min(5, len(dataset)), random_state=7)
    sample_options = ["Choose a sample..."] + [
        f"{row.label}: {row.text[:90]}" for row in samples.itertuples()
    ]
    selected_sample = st.selectbox("🧪 Sample news for testing", sample_options)
    if st.button("Use selected sample", disabled=selected_sample == sample_options[0]):
        st.session_state["news_text"] = selected_sample.split(": ", 1)[1]

    if st.button("🧹 Clear input"):
        st.session_state["news_text"] = ""
    news_text = st.text_area(
        "News headline or article",
        key="news_text",
        height=180,
        placeholder="Paste or type news text here...",
    )
    if st.button("🔍 Analyze news", type="primary"):
        if not news_text.strip():
            st.warning("Please enter a headline or article before analyzing.")
        else:
            cleaned = clean_text(news_text)
            if not cleaned:
                st.warning("No usable words remain after cleaning. Please enter more text.")
            else:
                features = vectorizer.transform([cleaned])
                prediction = str(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0]
                class_index = list(model.classes_).index(prediction)
                confidence = float(probabilities[class_index])
                if prediction == "REAL":
                    st.success("## ✅ REAL NEWS (model prediction)")
                else:
                    st.error("## 🚨 FAKE NEWS (model prediction)")
                st.metric("Model confidence", f"{confidence:.1%}")
                st.caption(
                    "Confidence is the classifier's estimated probability, "
                    "not proof that the story is true or false."
                )

elif page == "📊 Dataset Analysis":
    st.subheader("Explore the demonstration dataset")
    counts = dataset["label"].value_counts().reindex(["REAL", "FAKE"], fill_value=0)
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Total records", f"{len(dataset):,}")
    metric_b.metric("REAL", f"{int(counts['REAL']):,}")
    metric_c.metric("FAKE", f"{int(counts['FAKE']):,}")

    chart_col, length_col = st.columns(2)
    with chart_col:
        st.markdown("#### News records by label")
        st.bar_chart(counts.rename_axis("News label"))
    with length_col:
        st.markdown("#### Average word count")
        word_counts = dataset.assign(
            words=dataset["text"].str.split().str.len()
        ).groupby("label")["words"].mean()
        st.bar_chart(word_counts.reindex(["REAL", "FAKE"], fill_value=0))
    st.markdown("#### Dataset preview")
    st.dataframe(dataset, use_container_width=True, hide_index=True)
    st.caption(
        "The included CSV is a small, hand-written teaching sample; it is not "
        "a representative or independently verified news corpus."
    )

else:
    st.subheader("Model performance")
    if not metrics:
        st.warning("Evaluation metrics are missing. Run `python train_model.py` to create them.")
    else:
        st.caption(
            f"Metrics were calculated on {metrics['test_records']} held-out "
            f"records from a stratified train/test split (random state 42)."
        )
        columns = st.columns(4)
        for column, title, key in zip(
            columns,
            ["Accuracy", "Precision (FAKE)", "Recall (FAKE)", "F1-score (FAKE)"],
            ["accuracy", "precision", "recall", "f1_score"],
        ):
            column.metric(title, f"{metrics[key]:.1%}")
        report = metrics.get("classification_report", {})
        per_class = pd.DataFrame(
            {
                label: {
                    "Precision": report.get(label, {}).get("precision", 0.0),
                    "Recall": report.get(label, {}).get("recall", 0.0),
                    "F1-score": report.get(label, {}).get("f1-score", 0.0),
                }
                for label in ["REAL", "FAKE"]
            }
        )
        st.markdown("#### Per-class evaluation")
        st.bar_chart(per_class.T)
        st.caption(
            "A small sample can produce unstable scores. Metrics describe this "
            "particular holdout split, not expected real-world accuracy."
        )
