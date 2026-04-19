# SVM — Credit Card Fraud Detection Demo

An interactive Streamlit app that teaches Support Vector Machines using real credit
card fraud data. Built as a 30-minute classroom demo for an ML course.

## What's inside

- **`app.py`** — the interactive Streamlit demo. Covers the problem, margin, soft
  margin + C, kernel trick, polynomial + RBF kernels (with worked numeric examples),
  a full-feature model, and the SVM-to-fraud connection.
- **`svm_fraud_model.py`** — standalone training script. Trains an RBF SVM, prints
  metrics, saves a confusion matrix + ROC plot.
- **`DEMO_SCRIPT.md`** — 30-minute talk script for two presenters.
- **`requirements.txt`** — dependencies.

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset at `../archive/creditcard.csv` (Kaggle: *Credit Card Fraud Detection*).

## Run the app

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`.

## Run the standalone model

```bash
python svm_fraud_model.py
```

Prints evaluation metrics and saves `svm_fraud_results.png`.

## Notes

- Training is CPU-only (sklearn SVM). We subsample legit transactions to 6,000 to keep
  training under 10 seconds. Full 284K is doable but slow (SVM is O(n²–n³)).
- Class imbalance (~0.17% fraud) is handled via `class_weight='balanced'`.
- Evaluation prioritises **recall** and **ROC AUC** — accuracy is misleading here.
