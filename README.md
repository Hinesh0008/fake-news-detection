# Fake News Detection System Using Python and Machine Learning

A beginner-friendly Data Mining mini project that demonstrates how to prepare
news text, convert it into TF-IDF features, train a Logistic Regression
classifier, and explore predictions in a Streamlit dashboard.

> **Important limitation:** `dataset.csv` contains a small, hand-written
> teaching sample. It is not a representative or independently verified news
> corpus. The model may make mistakes, and its predictions are automated
> estimates—not verified facts, journalism, or evidence that a claim is true or
> false. Do not rely on it to make decisions or label real-world reporting.

## Features

- Four dashboard sections: Home, Fake News Prediction, Dataset Analysis, and
  Model Performance.
- Text cleaning (lowercase, URL/symbol removal, and English stopword removal
  while preserving negation words).
- TF-IDF unigram/bigram features and a Logistic Regression text classifier.
- REAL/FAKE prediction with the model's estimated class probability.
- Dataset counts, label charts, word-count chart, and CSV preview.
- Accuracy, precision, recall, and F1-score measured on a stratified holdout.
- Example news text and a clear-input button.
- Input and dataset validation with actionable error messages.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard and prediction interface |
| `train_model.py` | Validates the CSV, evaluates, trains, and saves artifacts |
| `preprocessing.py` | Shared text-cleaning function |
| `dataset.csv` | Small, balanced demonstration dataset (`text`, `label`) |
| `fake_news_model.pkl` | Trained Logistic Regression model (created by training) |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (created by training) |
| `model_metrics.json` | Holdout metrics (created by training) |
| `requirements.txt` | Python package dependencies |

## Installation and execution

Use Python 3.10 or newer. From this project folder, create and activate a
virtual environment:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your
browser. Run `python train_model.py` again whenever you change `dataset.csv`.
The generated pickle files must be trusted, since loading pickle files from
unknown sources can execute unsafe code.

## How the Data Mining workflow works

1. **Data collection:** records are read from `dataset.csv`. Each row has a
   `text` field and a `label` field (`REAL` or `FAKE`).
2. **Data cleaning:** `preprocessing.py` lowercases text, removes URLs and
   non-letter symbols, then filters common English stopwords while preserving
   negation words (`no`, `nor`, and `not`).
3. **Feature extraction:** `TfidfVectorizer` creates weighted word and
   two-word-phrase features.
4. **Classification:** Logistic Regression learns patterns from labeled text.
5. **Evaluation:** a stratified 75/25 train/test split is used to calculate
   accuracy, precision, recall, and F1-score. The final dashboard model is then
   fitted on all valid CSV rows.
6. **Prediction:** the app cleans new text with the same function and applies
   the fitted vectorizer and classifier.

## Extending the demo

For meaningful experiments, replace the demonstration rows with a properly
licensed, documented dataset whose labels and sources have been checked. Keep
the CSV column names (`text`, `label`) and ensure there are enough examples of
both classes. Re-run the training script and compare metrics on a sufficiently
large, representative, and preferably independently collected test set.
