"""Train and save the fake-news text classifier and evaluation results."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
LABELS = {"REAL", "FAKE"}


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Read the CSV and reject missing, empty, or unsupported records."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("dataset.csv must contain 'text' and 'label' columns.")

    data = data.loc[:, ["text", "label"]].dropna().copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.upper()
    data = data[data["text"].ne("")]
    invalid_labels = sorted(set(data["label"]) - LABELS)
    if invalid_labels:
        raise ValueError(f"Unsupported labels in dataset: {invalid_labels}")
    if data.empty:
        raise ValueError("The dataset has no valid, non-empty records.")
    if set(data["label"]) != LABELS:
        raise ValueError("The dataset must contain both REAL and FAKE records.")
    if data["label"].value_counts().min() < 2:
        raise ValueError("At least two records per label are required.")

    data["clean_text"] = data["text"].map(clean_text)
    data = data[data["clean_text"].ne("")]
    if data.empty or set(data["label"]) != LABELS:
        raise ValueError("The cleaned dataset must contain usable REAL and FAKE records.")
    if data["label"].value_counts().min() < 2:
        raise ValueError("At least two usable records per label are required.")
    return data.reset_index(drop=True)


def main() -> None:
    data = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )

    # Fit the text vocabulary only on training data before evaluating.
    evaluation_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=10000,
    )
    train_features = evaluation_vectorizer.fit_transform(x_train)
    test_features = evaluation_vectorizer.transform(x_test)
    evaluation_model = LogisticRegression(max_iter=1000, random_state=42)
    evaluation_model.fit(train_features, y_train)
    predictions = evaluation_model.predict(test_features)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label="FAKE", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label="FAKE", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, pos_label="FAKE", zero_division=0)
        ),
        "test_records": int(len(y_test)),
        "training_records": int(len(y_train)),
        "dataset_records": int(len(data)),
        "classification_report": classification_report(
            y_test, predictions, labels=["REAL", "FAKE"], zero_division=0,
            output_dict=True,
        ),
    }

    # Refit on all available rows for the model used by the dashboard.
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=10000,
    )
    all_features = vectorizer.fit_transform(data["clean_text"])
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(all_features, data["label"])

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Trained model using {len(data)} valid dataset records.")
    print(
        f"Holdout accuracy: {metrics['accuracy']:.3f} | "
        f"precision: {metrics['precision']:.3f} | "
        f"recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f}"
    )
    print(f"Saved model, vectorizer, and metrics in: {BASE_DIR}")


if __name__ == "__main__":
    main()
