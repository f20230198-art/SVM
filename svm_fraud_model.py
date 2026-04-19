"""
Credit Card Fraud Detection using Support Vector Machines.

Trains an RBF-kernel SVM on the Kaggle Credit Card Fraud dataset,
evaluates it with precision/recall/F1 and ROC-AUC, and prints a
confusion matrix. Handles severe class imbalance via class_weight.

Usage:
    python svm_fraud_model.py

Dataset expected at: ../archive/creditcard.csv (relative to this file).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "archive", "creditcard.csv")
N_LEGIT_SAMPLES = 6000   # subsample legit class for tractable SVM training
KERNEL = "rbf"
C_VALUE = 1.0
GAMMA = "scale"
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------------------
def load_and_prepare(path, n_legit=N_LEGIT_SAMPLES, seed=RANDOM_STATE):
    df = pd.read_csv(path)

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=n_legit, random_state=seed)
    data = pd.concat([fraud, legit]).sample(frac=1, random_state=seed).reset_index(drop=True)

    X = data.drop(columns=["Class"]).values
    y = data["Class"].values

    print(f"Dataset: {len(df):,} rows total")
    print(f"  Fraud cases: {(df['Class'] == 1).sum()}")
    print(f"  Legit cases: {(df['Class'] == 0).sum()}")
    print(f"  Fraud rate:  {(df['Class'] == 1).mean() * 100:.3f}%")
    print(f"Training subset: {len(data)} rows "
          f"({(y == 1).sum()} fraud + {(y == 0).sum()} legit)\n")
    return X, y


# ---------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------
def train_svm(X, y, kernel=KERNEL, C=C_VALUE, gamma=GAMMA, seed=RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight="balanced",
    )

    print(f"Training SVM (kernel={kernel}, C={C}, gamma={gamma})...")
    model.fit(X_train_s, y_train)
    print(f"  Support vectors: {model.support_vectors_.shape[0]}")
    print(f"  Training size:   {len(X_train)} / Test size: {len(X_test)}\n")

    return model, scaler, (X_test_s, y_test)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print("=" * 48)
    print("Test Results")
    print("=" * 48)
    print(f"Precision (fraud): {precision:.3f}")
    print(f"Recall    (fraud): {recall:.3f}")
    print(f"F1 score:          {f1:.3f}")
    print(f"ROC AUC:           {roc_auc:.3f}")
    print()
    print("Confusion matrix:")
    print(f"                 Pred Legit   Pred Fraud")
    print(f"  True Legit     {tn:10d}   {fp:10d}")
    print(f"  True Fraud     {fn:10d}   {tp:10d}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
    }


def cross_validate_C(X, y, C_values=(0.1, 1.0, 10.0), kernel=KERNEL, gamma=GAMMA, seed=RANDOM_STATE):
    """Pick the best C via 5-fold stratified cross-validation (F1 scoring)."""
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    print("5-fold cross-validation for C (scoring = F1)")
    print("-" * 48)
    scores = {}
    for C in C_values:
        model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced")
        cv_scores = cross_val_score(model, X_s, y, cv=cv, scoring="f1", n_jobs=-1)
        scores[C] = cv_scores.mean()
        print(f"  C = {C:>6}: F1 = {cv_scores.mean():.3f}  (std {cv_scores.std():.3f})")
    best_C = max(scores, key=scores.get)
    print(f"Best C: {best_C}\n")
    return best_C


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_results(metrics, save_path="svm_fraud_results.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Legit", "Pred Fraud"],
                yticklabels=["True Legit", "True Fraud"], ax=axes[0])
    axes[0].set_title("Confusion Matrix")

    axes[1].plot(metrics["fpr"], metrics["tpr"], color="#2F7FBF", lw=2,
                 label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="#aaa")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Plot saved to: {save_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    X, y = load_and_prepare(DATA_PATH)

    # Optional: find best C via CV (uncomment to enable — adds ~30 s)
    # best_C = cross_validate_C(X, y, C_values=(0.1, 1.0, 10.0))

    model, scaler, (X_test, y_test) = train_svm(X, y)
    metrics = evaluate(model, X_test, y_test)
    plot_results(metrics)


if __name__ == "__main__":
    main()
