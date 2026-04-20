# SVM — Credit Card Fraud Detection

A 30-minute code walkthrough of Support Vector Machines applied to credit-card fraud detection, built around the Kaggle *Credit Card Fraud Detection* dataset.

The focus is the **code**: each stage of the ML pipeline is shown, run, and explained line by line inside a Jupyter notebook.

## Files

- **`svm_walkthrough.ipynb`** — the main deliverable. A notebook that loads the data, trains an RBF-kernel SVM, evaluates it with precision/recall/F1/ROC-AUC, and visualizes the results. Markdown cells between code cells explain what every block does and why.
- **`svm_fraud_model.py`** — the same pipeline as a standalone Python script. Run it to reproduce all results end-to-end from the command line.
- **`requirements.txt`** — Python dependencies.
- **`archive/creditcard.csv`** — dataset (from Kaggle).

## Setup

```
pip install -r requirements.txt
```

Download `creditcard.csv` from the Kaggle *Credit Card Fraud Detection* dataset and place it at `archive/creditcard.csv`.

## Run the notebook

```
jupyter notebook svm_walkthrough.ipynb
```

Or open it directly in VS Code and run all cells.

## Run the standalone script

```
python svm_fraud_model.py
```

Prints metrics to the console and saves a confusion matrix + ROC curve to `svm_fraud_results.png`.

## What the code does

1. Loads ~284k transactions, keeps all ~492 fraud cases, subsamples 6,000 legit cases (SVM is slow on large N).
2. Splits into train/test with stratification so both sides keep the same fraud ratio.
3. Applies `StandardScaler` — mandatory for SVM with RBF kernel.
4. Trains `SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')`. The `class_weight='balanced'` argument is what makes it work on imbalanced fraud data.
5. Evaluates with precision, recall, F1, and ROC-AUC (accuracy is misleading at 0.17% fraud rate).
6. Plots the confusion matrix and ROC curve.
7. Optionally tunes `C` via 5-fold stratified cross-validation.
