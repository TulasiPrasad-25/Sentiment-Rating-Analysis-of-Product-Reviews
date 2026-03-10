"""
FastAPI - AI Product Review System
===================================
Endpoints:
  POST /train            - Load CSV, preprocess, train Logistic Regression & LSTM
  POST /predict/sentiment - Predict sentiment for a review text
  POST /predict/rating    - Predict star rating (1-5) for a review text
  POST /urgency           - Compute urgency/complaint score for a review text
  GET  /eda/stats         - Basic dataset statistics
  GET  /eda/top-words     - Top N most-common words in the corpus
  GET  /model/comparison  - LR vs LSTM performance metrics

Run with:
    uvicorn main:app --reload
"""

import re
import string
import collections
import io
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import nltk
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── keras / tensorflow ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress TF noise
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK bootstrap
# ─────────────────────────────────────────────────────────────────────────────
for pkg, path in [("stopwords", "corpora/stopwords"),
                  ("punkt", "tokenizers/punkt"),
                  ("punkt_tab", "tokenizers/punkt_tab")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(nltk.corpus.stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_MAPPING = {"positive": 2, "neutral": 1, "negative": 0}
REVERSE_SENTIMENT = {v: k for k, v in SENTIMENT_MAPPING.items()}

URGENCY_KEYWORDS = [
    "urgent", "problem", "bad", "disappointed", "issue", "faulty", "return",
    "broken", "never again", "worst", "poor quality", "defective", "unresponsive",
    "terrible", "regret", "complaint", "dissatisfied", "unhappy", "frustrated",
    "horrible", "waste of money", "beware", "scam", "useless",
]

MAX_WORDS = 10_000
MAX_LEN   = 100
TFIDF_MAX = 5_000

# ─────────────────────────────────────────────────────────────────────────────
# In-memory state  (populated after /train is called)
# ─────────────────────────────────────────────────────────────────────────────
state: dict = {
    "df": None,
    "tfidf_vectorizer": None,
    "lr_sentiment_model": None,
    "lr_rating_model": None,
    "lstm_model": None,
    "keras_tokenizer": None,
    "lr_sentiment_report": None,
    "lstm_report": None,
    "trained": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Lowercase → remove punctuation → tokenise → remove stopwords."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


def calculate_urgency_score(review: str) -> int:
    """Count urgency/complaint keyword occurrences in review text."""
    if not isinstance(review, str):
        return 0
    review_lower = review.lower()
    score = 0
    for kw in URGENCY_KEYWORDS:
        score += len(re.findall(r"\b" + re.escape(kw) + r"\b", review_lower))
    return score


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the dataset CSV and apply the same cleaning as the notebook."""
    df = pd.read_csv(csv_path)

    df["Review"]  = df["Review"].fillna("")
    df["Summary"] = df["Summary"].fillna("")

    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    mode_rate = df["Rate"].mode()[0]
    df["Rate"] = df["Rate"].fillna(mode_rate).astype(int)

    df["review_length"]  = df["Review"].apply(len)
    df["summary_length"] = df["Summary"].apply(len)

    df["Sentiment_numeric"] = df["Sentiment"].map(SENTIMENT_MAPPING)
    df.dropna(subset=["Sentiment_numeric"], inplace=True)
    df["Sentiment_numeric"] = df["Sentiment_numeric"].astype(int)

    df["processed_review"] = df["Review"].apply(preprocess_text)
    df["urgency_complaint_score"] = df["Review"].apply(calculate_urgency_score)

    return df


def build_tfidf(df: pd.DataFrame):
    vec = TfidfVectorizer(max_features=TFIDF_MAX)
    X   = vec.fit_transform(df["processed_review"])
    return vec, X


def train_lr_sentiment(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    report = classification_report(
        y_te, y_pred,
        target_names=[REVERSE_SENTIMENT[i] for i in sorted(REVERSE_SENTIMENT)],
        zero_division=0, output_dict=True,
    )
    return model, report


def train_lr_rating(X, y_rate):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_rate, test_size=0.2, random_state=42, stratify=y_rate
    )
    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_tr, y_tr)
    return model


def build_and_train_lstm(df: pd.DataFrame):
    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tok.fit_on_texts(df["processed_review"])
    seqs   = tok.texts_to_sequences(df["processed_review"])
    X_pad  = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    y      = df["Sentiment_numeric"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_pad, y, test_size=0.2, random_state=42, stratify=y
    )

    vocab_size = len(tok.word_index) + 1
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100),
        LSTM(128),
        Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=5, batch_size=64,
              validation_split=0.2, verbose=0)

    y_pred_prob = model.predict(X_te, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    target_names = [REVERSE_SENTIMENT[i] for i in sorted(REVERSE_SENTIMENT)]
    report = classification_report(
        y_te, y_pred,
        target_names=target_names,
        zero_division=0, output_dict=True,
    )
    return tok, model, report


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    csv_path: str = "Dataset-SA.csv"
    train_lstm: bool = False       # LSTM is optional – it takes a few minutes


class ReviewRequest(BaseModel):
    review: str


class TopWordsRequest(BaseModel):
    n: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Product Review System API",
    description="Sentiment analysis, rating prediction & urgency scoring for e-commerce reviews.",
    version="1.0.0",
)


def _require_trained():
    if not state["trained"]:
        raise HTTPException(
            status_code=400,
            detail="Models not trained yet. Call POST /train first.",
        )


# ── /train ────────────────────────────────────────────────────────────────────

@app.post("/train", summary="Load dataset, preprocess and train models")
def train(req: TrainRequest):
    """
    - Loads the CSV from `csv_path`
    - Cleans & preprocesses data (mirrors the notebook pipeline)
    - Trains **Logistic Regression** for sentiment & rating prediction
    - Optionally trains the **LSTM** model (set `train_lstm=true`)

    Returns accuracy metrics for each trained model.
    """
    if not os.path.exists(req.csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"CSV not found at '{req.csv_path}'. "
                   "Download from: https://drive.google.com/file/d/1W-88FKdbZ3BMzgz1HWa-RKk4t5OWPuji/view",
        )

    logger.info("Loading and cleaning dataset …")
    df = load_and_clean(req.csv_path)

    logger.info("Building TF-IDF features …")
    vec, X_tfidf = build_tfidf(df)

    logger.info("Training LR sentiment model …")
    lr_s, lr_s_report = train_lr_sentiment(X_tfidf, df["Sentiment_numeric"])

    logger.info("Training LR rating model …")
    lr_r = train_lr_rating(X_tfidf, df["Rate"])

    # Store in global state
    state["df"]                  = df
    state["tfidf_vectorizer"]    = vec
    state["lr_sentiment_model"]  = lr_s
    state["lr_rating_model"]     = lr_r
    state["lr_sentiment_report"] = lr_s_report

    lstm_result = None
    if req.train_lstm:
        logger.info("Training LSTM model (this may take a few minutes) …")
        tok, lstm_model, lstm_report = build_and_train_lstm(df)
        state["keras_tokenizer"] = tok
        state["lstm_model"]      = lstm_model
        state["lstm_report"]     = lstm_report
        lstm_result = {
            "accuracy":          lstm_report["accuracy"],
            "weighted_avg_f1":   lstm_report["weighted avg"]["f1-score"],
        }

    state["trained"] = True
    logger.info("Training complete.")

    return {
        "status":  "success",
        "rows_loaded": len(df),
        "logistic_regression_sentiment": {
            "accuracy":        lr_s_report["accuracy"],
            "weighted_avg_f1": lr_s_report["weighted avg"]["f1-score"],
        },
        "lstm": lstm_result,
    }


# ── /predict/sentiment ────────────────────────────────────────────────────────

@app.post("/predict/sentiment", summary="Predict sentiment for a review")
def predict_sentiment(req: ReviewRequest):
    """
    Returns predicted sentiment (`positive` / `neutral` / `negative`) and
    class probabilities using the Logistic Regression model.
    """
    _require_trained()

    processed = preprocess_text(req.review)
    vec       = state["tfidf_vectorizer"]
    model     = state["lr_sentiment_model"]

    X       = vec.transform([processed])
    label   = int(model.predict(X)[0])
    probs   = model.predict_proba(X)[0].tolist()
    classes = [REVERSE_SENTIMENT[c] for c in model.classes_]

    return {
        "review":     req.review,
        "sentiment":  REVERSE_SENTIMENT[label],
        "probabilities": dict(zip(classes, [round(p, 4) for p in probs])),
    }


# ── /predict/rating ───────────────────────────────────────────────────────────

@app.post("/predict/rating", summary="Predict star rating (1-5) for a review")
def predict_rating(req: ReviewRequest):
    """
    Returns the predicted star rating (1–5) and class probabilities
    using the Logistic Regression model trained on `Rate`.
    """
    _require_trained()

    processed = preprocess_text(req.review)
    vec   = state["tfidf_vectorizer"]
    model = state["lr_rating_model"]

    X          = vec.transform([processed])
    rating     = int(model.predict(X)[0])
    probs      = model.predict_proba(X)[0].tolist()
    classes    = [str(c) for c in model.classes_]

    return {
        "review":         req.review,
        "predicted_rating": rating,
        "probabilities":  dict(zip(classes, [round(p, 4) for p in probs])),
    }


# ── /predict/sentiment/lstm ───────────────────────────────────────────────────

@app.post("/predict/sentiment/lstm",
          summary="Predict sentiment using the LSTM model")
def predict_sentiment_lstm(req: ReviewRequest):
    """
    Requires LSTM to have been trained (`train_lstm=true` in `/train`).
    Returns predicted sentiment and class probabilities.
    """
    _require_trained()
    if state["lstm_model"] is None:
        raise HTTPException(
            status_code=400,
            detail="LSTM model not trained. Call POST /train with train_lstm=true.",
        )

    processed = preprocess_text(req.review)
    tok       = state["keras_tokenizer"]
    model     = state["lstm_model"]

    seq   = tok.texts_to_sequences([processed])
    X_pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    probs  = model.predict(X_pad, verbose=0)[0]
    label  = int(np.argmax(probs))
    names  = [REVERSE_SENTIMENT[i] for i in range(3)]   # 0=neg,1=neu,2=pos

    return {
        "review":    req.review,
        "sentiment": REVERSE_SENTIMENT[label],
        "probabilities": {n: round(float(p), 4) for n, p in zip(names, probs)},
    }


# ── /urgency ─────────────────────────────────────────────────────────────────

@app.post("/urgency", summary="Compute urgency/complaint score for a review")
def urgency(req: ReviewRequest):
    """
    Scans the review for predefined urgency/complaint keywords and returns
    a score (0 = no urgency keywords found).
    """
    score    = calculate_urgency_score(req.review)
    keywords = []
    for kw in URGENCY_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", req.review.lower()):
            keywords.append(kw)

    return {
        "review":                req.review,
        "urgency_complaint_score": score,
        "matched_keywords":      keywords,
        "is_urgent":             score > 0,
    }


# ── /eda/stats ────────────────────────────────────────────────────────────────

@app.get("/eda/stats", summary="Basic dataset statistics")
def eda_stats():
    """Returns shape, sentiment distribution, rating distribution and
    review-length statistics for the loaded dataset."""
    _require_trained()
    df = state["df"]

    return {
        "total_rows": len(df),
        "sentiment_distribution": df["Sentiment"].value_counts().to_dict(),
        "rating_distribution":    df["Rate"].value_counts().sort_index().to_dict(),
        "review_length_stats": {
            "mean":   round(df["review_length"].mean(), 2),
            "median": df["review_length"].median(),
            "max":    int(df["review_length"].max()),
            "min":    int(df["review_length"].min()),
        },
        "urgency_score_stats": {
            "mean":            round(df["urgency_complaint_score"].mean(), 4),
            "reviews_with_urgency": int((df["urgency_complaint_score"] > 0).sum()),
        },
    }


# ── /eda/top-words ────────────────────────────────────────────────────────────

@app.get("/eda/top-words", summary="Most common words in the review corpus")
def top_words(n: int = 20):
    """Returns the `n` most common words found across all reviews."""
    _require_trained()
    df = state["df"]

    combined = " ".join(df["Review"].astype(str).tolist()).lower()
    words    = re.findall(r"\b[a-z]{3,}\b", combined)
    counts   = collections.Counter(words).most_common(n)

    return {
        "top_words": [{"word": w, "count": c} for w, c in counts],
    }


# ── /model/comparison ────────────────────────────────────────────────────────

@app.get("/model/comparison", summary="Compare LR vs LSTM performance metrics")
def model_comparison():
    """Returns a side-by-side comparison of Logistic Regression and LSTM
    models (accuracy, macro-avg F1, weighted-avg F1)."""
    _require_trained()

    lr  = state["lr_sentiment_report"]
    lm  = state["lstm_report"]

    result = {
        "logistic_regression": {
            "accuracy":          round(lr["accuracy"], 4),
            "macro_avg_f1":      round(lr["macro avg"]["f1-score"], 4),
            "weighted_avg_f1":   round(lr["weighted avg"]["f1-score"], 4),
            "per_class":         {k: v for k, v in lr.items()
                                  if k not in ("accuracy", "macro avg", "weighted avg")},
        },
        "lstm": None,
    }

    if lm:
        result["lstm"] = {
            "accuracy":          round(lm["accuracy"], 4),
            "macro_avg_f1":      round(lm["macro avg"]["f1-score"], 4),
            "weighted_avg_f1":   round(lm["weighted avg"]["f1-score"], 4),
            "per_class":         {k: v for k, v in lm.items()
                                  if k not in ("accuracy", "macro avg", "weighted avg")},
        }

    return result


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "models_trained": state["trained"]}
