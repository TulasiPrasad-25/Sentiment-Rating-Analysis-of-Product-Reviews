# 🛍️ AI Product Review System

An NLP-powered REST API built with **FastAPI** that performs sentiment analysis, star rating prediction, and urgency/complaint scoring on e-commerce product reviews.

> Built on a dataset of 200,000+ real product reviews using Logistic Regression and an optional LSTM deep learning model.

---

## 📌 Features

- 🔍 **Sentiment Prediction** — Classifies reviews as `positive`, `neutral`, or `negative`
- ⭐ **Rating Prediction** — Predicts star rating (1–5) from review text
- 🚨 **Urgency Scoring** — Detects complaint/urgency keywords and flags critical reviews
- 📊 **EDA Endpoints** — Dataset statistics, rating distributions, top words
- 🤖 **Dual Models** — Logistic Regression (fast) + LSTM (deep learning, optional)
- 📖 **Auto Swagger Docs** — Interactive API docs out of the box

---

## 🗂️ Project Structure

```
your-repo/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dataset-SA.csv       # Dataset (download separately — see below)
└── README.md
```

---

## ⚙️ Prerequisites

- Python **3.10** or **3.11**
- pip
- Git

Check your Python version:
```bash
python --version
```

---

## 🚀 Local Setup & Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2. Download the Dataset

The dataset is not included in the repo due to its size (~200K rows).

👉 Download it here: [Dataset-SA.csv on Google Drive](https://drive.google.com/file/d/1W-88FKdbZ3BMzgz1HWa-RKk4t5OWPuji/view?usp=sharing)

After downloading, place the file in the **root of the project folder**:

```
your-repo/
└── Dataset-SA.csv   ← here
```

---

### 3. Create a Virtual Environment

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> This installs FastAPI, TensorFlow, scikit-learn, NLTK, pandas, and all other required packages. May take 2–5 minutes on first run.

---

### 5. Start the Server

```bash
uvicorn main:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

---

### 6. Open the Interactive API Docs

Visit in your browser:
```
http://127.0.0.1:8000/docs
```

This opens the **Swagger UI** where you can test all endpoints interactively.

---

### 7. Train the Models (Required First Step)

Before using any prediction endpoint, you must train the models.

**In Swagger UI:**
1. Click `POST /train`
2. Click **Try it out**
3. Use this request body:

```json
{
  "csv_path": "Dataset-SA.csv",
  "train_lstm": false
}
```

4. Click **Execute**

> ⏱️ Training takes **1–3 minutes**. Set `train_lstm: true` to also train the LSTM model (adds ~5–10 minutes).

A successful response looks like:
```json
{
  "status": "success",
  "rows_loaded": 205052,
  "logistic_regression_sentiment": {
    "accuracy": 0.8952,
    "weighted_avg_f1": 0.87
  },
  "lstm": null
}
```

---

## 📡 API Endpoints

### `POST /train`
Load dataset, preprocess, and train models.

```json
{
  "csv_path": "Dataset-SA.csv",
  "train_lstm": false
}
```

---

### `POST /predict/sentiment`
Predict sentiment of a review using Logistic Regression.

```json
{ "review": "This product is absolutely amazing, works perfectly!" }
```

Response:
```json
{
  "review": "This product is absolutely amazing, works perfectly!",
  "sentiment": "positive",
  "probabilities": {
    "negative": 0.03,
    "neutral": 0.05,
    "positive": 0.92
  }
}
```

---

### `POST /predict/sentiment/lstm`
Predict sentiment using the LSTM model.
> Requires `train_lstm: true` during `/train`.

```json
{ "review": "Stopped working after two days, very disappointed." }
```

---

### `POST /predict/rating`
Predict star rating (1–5) from review text.

```json
{ "review": "Decent product but packaging was damaged on arrival." }
```

Response:
```json
{
  "review": "Decent product but packaging was damaged on arrival.",
  "predicted_rating": 3,
  "probabilities": {
    "1": 0.08, "2": 0.12, "3": 0.45, "4": 0.25, "5": 0.10
  }
}
```

---

### `POST /urgency`
Compute urgency/complaint score and detect keywords.

```json
{ "review": "Terrible quality, complete scam, I want a return!" }
```

Response:
```json
{
  "review": "Terrible quality, complete scam, I want a return!",
  "urgency_complaint_score": 3,
  "matched_keywords": ["terrible", "scam", "return"],
  "is_urgent": true
}
```

---

### `GET /eda/stats`
Returns dataset-level statistics — sentiment distribution, rating distribution, review length stats.

---

### `GET /eda/top-words?n=20`
Returns the top N most common words across all reviews.

---

### `GET /model/comparison`
Side-by-side performance comparison of Logistic Regression vs LSTM.

---

### `GET /health`
Simple health check.

```json
{ "status": "ok", "models_trained": true }
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|--------|-----|
| `CSV not found` | Make sure `Dataset-SA.csv` is in the same folder as `main.py` |
| `Models not trained yet` | Always call `POST /train` before any prediction endpoint |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` with venv activated |
| `Port already in use` | Run `uvicorn main:app --reload --port 8001` and open `localhost:8001/docs` |
| LSTM endpoint error | Train with `"train_lstm": true` before calling `/predict/sentiment/lstm` |
| Slow training | Normal — LR takes ~1–3 min, LSTM takes ~5–10 min on CPU |

---

## 🧠 Models Used

| Model | Task | Accuracy |
|-------|------|----------|
| Logistic Regression (TF-IDF) | Sentiment (3-class) | ~89.5% |
| Logistic Regression (TF-IDF) | Rating (1–5) | ~varies |
| LSTM (Embedding + 128 units) | Sentiment (3-class) | ~81.2% |

> Both models struggle with the `neutral` class due to class imbalance in the dataset. Future improvements: SMOTE oversampling, focal loss, or a transformer-based model.

---

## 📦 Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [scikit-learn](https://scikit-learn.org/) — Logistic Regression, TF-IDF
- [TensorFlow / Keras](https://www.tensorflow.org/) — LSTM model
- [NLTK](https://www.nltk.org/) — Text preprocessing
- [pandas](https://pandas.pydata.org/) — Data manipulation
- [Uvicorn](https://www.uvicorn.org/) — ASGI server

---

## 📄 License

MIT License — free to use and modify.
