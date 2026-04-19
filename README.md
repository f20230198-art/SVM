# SVM — Credit Card Fraud Detection

Interactive Streamlit app explaining Support Vector Machines using the Kaggle
Credit Card Fraud Detection dataset. Built as a 30-minute classroom presentation
for an ML course.

## Files

- **`app.py`** — Streamlit app. Covers the problem, maximal and soft margin, kernel
  trick, polynomial and RBF kernels (with worked numeric examples), full-feature
  training and evaluation, and mapping of SVM concepts to the fraud detection task.
- **`svm_fraud_model.py`** — Standalone training script. Trains an RBF SVM, prints
  metrics, and saves a confusion matrix and ROC plot to `svm_fraud_results.png`.
- **`requirements.txt`** — Python dependencies.

## Setup

```
pip install -r requirements.txt
```

Download `creditcard.csv` from the Kaggle *Credit Card Fraud Detection* dataset
and place it at `archive/creditcard.csv`.

## Run the app

```
python -m streamlit run app.py
```

Opens at `http://localhost:8501`.

## Run the standalone script

```
python svm_fraud_model.py
```

## Notes

- SVM training is CPU-only (sklearn). Legit transactions are subsampled to 6,000
  to keep training under 10 seconds.
- Class imbalance (~0.17% fraud) is handled via `class_weight='balanced'`.
- Primary metrics are recall and ROC AUC. Accuracy is misleading on this dataset.
