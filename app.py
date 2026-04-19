"""
SVM Credit Card Fraud Detection — Interactive Teaching Demo
Run with: streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA

# ----------------------------- Page config -----------------------------
st.set_page_config(
    page_title="SVM • Credit Card Fraud Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- Styling --------------------------------
ACCENT = "#2F7FBF"
FRAUD = "#D9534F"
LEGIT = "#4A90A4"

st.markdown(
    f"""
    <style>
    .main .block-container {{ max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem; }}
    h1, h2, h3 {{ color: #1f2d3d; }}
    .concept-box {{
        border-left: 4px solid {ACCENT};
        padding: 0.6rem 1rem;
        margin: 0.5rem 0 1rem 0;
        font-weight: 600;
    }}
    .key-idea {{
        border-left: 4px solid #e0a800;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }}
    .section-divider {{
        border-top: 1px solid #e0e6ed;
        margin: 3rem 0 2rem 0;
    }}
    .stPlotlyChart, .stPyplot {{ margin: 0 auto; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Data loading ----------------------------
@st.cache_data(show_spinner=True)
def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "archive", "creditcard.csv")
    df = pd.read_csv(path)
    return df

@st.cache_data
def prepare_subsample(df, n_legit=3000, n_fraud=None, seed=42):
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=n_legit, random_state=seed)
    if n_fraud is not None:
        fraud = fraud.sample(n=min(n_fraud, len(fraud)), random_state=seed)
    sub = pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sub

@st.cache_data
def pick_two_features(df, feat_a="V14", feat_b="V17"):
    X = df[[feat_a, feat_b]].values
    y = df["Class"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y

# ----------------------------- Plot helpers ----------------------------
def plot_decision_boundary(ax, model, X, y, title="", show_support=True, pad=0.6, grid=250):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid), np.linspace(y_min, y_max, grid))
    try:
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#e7f0f8", "#fbe7e7"], alpha=0.6)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"],
                   colors=["#888", ACCENT, "#888"], linewidths=[1, 1.8, 1])
    except Exception:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")

    legit_mask = y == 0
    fraud_mask = y == 1
    ax.scatter(X[legit_mask, 0], X[legit_mask, 1], c=LEGIT, s=16, alpha=0.6, label="Legit (0)", edgecolors="none")
    ax.scatter(X[fraud_mask, 0], X[fraud_mask, 1], c=FRAUD, s=28, alpha=0.85, label="Fraud (1)",
               edgecolors="white", linewidth=0.5)

    if show_support and hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", edgecolors="black",
                   linewidths=1.2, label="Support vectors")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Feature 1 (standardized)")
    ax.set_ylabel("Feature 2 (standardized)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def style_fig(fig):
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
    return fig


# =======================================================================
# HERO
# =======================================================================
st.title("Support Vector Machines — Credit Card Fraud Detection")
st.markdown(
    "Topics: maximal margin classifier, soft margin, kernel trick, polynomial kernel, "
    "RBF kernel, evaluation on an imbalanced dataset."
)
st.caption("Dataset: Kaggle *Credit Card Fraud Detection* — 284,807 transactions, 492 fraud.")

# Load data once
with st.spinner("Loading dataset..."):
    df = load_data()

# =======================================================================
# SECTION 1: THE PROBLEM
# =======================================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("1 · The Problem")

col1, col2 = st.columns([1.1, 1])
with col1:
    st.markdown(
        """
        Binary classification: fraud (1) vs legit (0).

        - 30 features: `V1`–`V28` (PCA-transformed for privacy), `Amount`, `Time`.
        - Extreme class imbalance: fraud < 0.2% of transactions.
        - Overlap in feature space: fraud is not cleanly separable from legit.
        """
    )

with col2:
    class_counts = df["Class"].value_counts()
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    bars = ax.bar(["Legit (0)", "Fraud (1)"], class_counts.values, color=[LEGIT, FRAUD])
    for bar, val in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}",
                ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Class imbalance")
    style_fig(fig)
    st.pyplot(fig)

st.markdown(
    f"""
    <div class="key-idea">
    Fraud rate = <b>{class_counts[1] / class_counts.sum() * 100:.3f}%</b>. A trivial classifier
    that predicts "not fraud" scores 99.83% accuracy and detects zero fraud. Accuracy is
    not a valid metric on this dataset.
    </div>
    """,
    unsafe_allow_html=True,
)

# =======================================================================
# SECTION 2: A SINGLE LINE BETWEEN TWO GROUPS
# =======================================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("2 · Separating Two Classes with a Hyperplane")

st.markdown(
    """
    <div class="concept-box">
    A hyperplane separates the two classes. SVM chooses the one with the <b>largest
    margin</b> — the widest empty region around it. The training points on the margin
    edges are the <b>support vectors</b>; they alone determine the hyperplane.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Linearly separable example (apples vs oranges by weight and diameter)")

rng = np.random.default_rng(7)
apples = rng.normal(loc=[150, 7], scale=[10, 0.4], size=(25, 2))
oranges = rng.normal(loc=[200, 9], scale=[12, 0.4], size=(25, 2))
X_toy = np.vstack([apples, oranges])
y_toy = np.array([0] * 25 + [1] * 25)
toy_model = SVC(kernel="linear", C=1.0).fit(X_toy, y_toy)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
x_min, x_max = X_toy[:, 0].min() - 10, X_toy[:, 0].max() + 10
y_min, y_max = X_toy[:, 1].min() - 1, X_toy[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = toy_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"],
           colors=["#888", ACCENT, "#888"], linewidths=[1, 1.8, 1])
ax.scatter(apples[:, 0], apples[:, 1], c="#D9534F", s=60, label="Apples", edgecolors="white")
ax.scatter(oranges[:, 0], oranges[:, 1], c="#F0A048", s=60, label="Oranges", edgecolors="white")
sv = toy_model.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=130, facecolors="none", edgecolors="black", linewidths=1.3,
           label="Support vectors")
ax.set_xlabel("Weight (g)")
ax.set_ylabel("Diameter (cm)")
ax.set_title("Two well-separated classes → one clean line")
ax.legend(fontsize=8, loc="lower right")
style_fig(fig)
_t1, _t2, _t3 = st.columns([1, 2, 1])
with _t2:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    """
    - Solid line: the decision boundary.
    - Dashed lines: the margin edges.
    - Circled points: the support vectors.

    Deleting any non-support-vector point leaves the boundary unchanged.
    """
)

st.markdown("#### Credit card data — features V14 and V17")

sub = prepare_subsample(df, n_legit=2000, n_fraud=None)
X2, y2 = pick_two_features(sub, "V14", "V17")

fig, ax = plt.subplots(figsize=(5.5, 4))
linear_demo = SVC(kernel="linear", C=1.0).fit(X2, y2)
plot_decision_boundary(ax, linear_demo, X2, y2,
                       title="Linear SVM on 2 features — margin boundaries dashed")
style_fig(fig)
_c1, _c2, _c3 = st.columns([1, 2, 1])
with _c2:
    st.pyplot(fig, use_container_width=True)

st.markdown("#### Optimization")
st.markdown(
    """
    A hyperplane is defined by **w · x + b = 0**, where **w** is normal to the plane and
    **b** is the offset. The margin width equals **2 / ‖w‖**. SVM solves:
    """
)
st.latex(r"\min_{\mathbf{w},\,b}\; \tfrac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to}\quad y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \geq 1 \;\; \forall i")
st.markdown("The solution **w** is a linear combination of the support vectors only.")

# =======================================================================
# SECTION 3: HARD MARGIN -> SOFT MARGIN
# =======================================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("3 · Soft Margin and the Parameter C")

st.markdown(
    """
    A **hard margin** requires all points to lie strictly outside the margin. This is
    only feasible for linearly separable data.

    A **soft margin** permits margin violations. The parameter **C** controls the
    tradeoff between margin width and violation penalty:

    - Large C → violations are costly → narrow margin, low bias, high variance.
    - Small C → violations are cheap → wider margin, higher bias, lower variance.
    """
)

st.latex(r"\min_{\mathbf{w},\,b} \;\; \tfrac{1}{2}\|\mathbf{w}\|^2 \;+\; C \sum_{i=1}^{n} \xi_i")

st.markdown(
    """
    Symbols:
    - **w, b** — define the hyperplane (w · x + b = 0).
    - **ξᵢ** — slack for point *i*. ξᵢ = 0: outside margin. 0 < ξᵢ ≤ 1: inside margin,
      correct side. ξᵢ > 1: misclassified.
    - **C** — penalty coefficient on total slack.

    The objective minimizes margin width and total slack, weighted by C.
    """
)

st.markdown("#### Selecting C via k-fold Cross-Validation")
st.markdown(
    """
    C cannot be chosen from training performance (large C always minimizes training
    error by fitting tightly). Standard procedure — **5-fold cross-validation**:

    1. Partition the training set into 5 equal folds.
    2. For each candidate C: train on 4 folds, evaluate on the held-out fold, repeat 5 times.
    3. Average the 5 scores. Select the C with the highest average.

    The same procedure selects γ (RBF kernel) and d (polynomial kernel).
    """
)

st.markdown("#### Interactive — vary C")

c_val = st.slider("C (regularization strength)", 0.01, 50.0, 1.0, 0.01, key="c_slider")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
soft = SVC(kernel="linear", C=c_val).fit(X2, y2)
plot_decision_boundary(axes[0], soft, X2, y2, title=f"Soft margin — C = {c_val:.2f}")

# Fixed comparison: very small C vs very large C
very_small = SVC(kernel="linear", C=0.01).fit(X2, y2)
plot_decision_boundary(axes[1], very_small, X2, y2, title="Reference: C = 0.01 (very soft, wide margin)")
style_fig(fig)
_sm1, _sm2, _sm3 = st.columns([1, 4, 1])
with _sm2:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    f"""
    <div class="key-idea">
    At C = {c_val:.2f}: {len(soft.support_vectors_)} support vectors out of {len(X2)} points.
    Lower C widens the margin and increases the number of support vectors.
    </div>
    """,
    unsafe_allow_html=True,
)

# =======================================================================
# SECTION 4: WHEN A LINE ISN'T ENOUGH
# =======================================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("4 · The Kernel Trick")

st.markdown(
    """
    When data is not linearly separable, map it into a higher-dimensional feature space
    where a linear separator exists.

    Example: a drug that cures at medium dosages and fails at low and high dosages is not
    separable on the dosage axis alone. Adding a dosage² axis makes the classes linearly
    separable.
    """
)

# Synthetic 1D dosage demo
dos_x = np.linspace(-3, 3, 80)
dos_y_class = ((dos_x > -1.2) & (dos_x < 1.2)).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
# 1D view
axes[0].scatter(dos_x, np.zeros_like(dos_x), c=[FRAUD if c else LEGIT for c in dos_y_class],
                s=40, edgecolors="white")
axes[0].set_yticks([])
axes[0].set_xlabel("Dosage")
axes[0].set_title("1D: no single threshold separates the classes")
axes[0].axhline(0, color="#ccc", linewidth=0.8)

# Lifted to 2D with x^2
axes[1].scatter(dos_x, dos_x**2, c=[FRAUD if c else LEGIT for c in dos_y_class],
                s=60, edgecolors="white")
axes[1].axhline(1.5, color=ACCENT, linestyle="--", linewidth=1.5, label="Linear separator in lifted space")
axes[1].set_xlabel("Dosage")
axes[1].set_ylabel("Dosage²")
axes[1].set_title("2D after lifting: a straight line works")
axes[1].legend()
style_fig(fig)
_d1, _d2, _d3 = st.columns([1, 3, 1])
with _d2:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    """
    <div class="concept-box">
    SVM's optimization depends on the data only through dot products between pairs of
    points. A <b>kernel function</b> K(a, b) returns the dot product of a and b in the
    higher-dimensional space without explicitly computing the mapping. This is the
    <b>kernel trick</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Polynomial kernel --------------------
st.subheader("4a · Polynomial Kernel")
st.latex(r"K(\mathbf{a}, \mathbf{b}) = (\mathbf{a} \cdot \mathbf{b} + r)^d")
st.markdown(
    """
    - **d** — polynomial degree. d = 1 is linear; d ≥ 2 produces curved boundaries.
    - **r** — constant offset. When the polynomial is expanded, its terms correspond to
      the coordinates of the implicit higher-dimensional feature space.
    """
)

st.markdown("#### Worked example (d = 2, r = 1)")
st.markdown("Let **a = (1, 2)**, **b = (3, 4)**. Compute the dot product:")
st.latex(r"\mathbf{a} \cdot \mathbf{b} = (1)(3) + (2)(4) = 3 + 8 = 11")
st.markdown("Apply the kernel:")
st.latex(r"K(\mathbf{a}, \mathbf{b}) = (11 + 1)^2 = 12^2 = 144")
st.markdown(
    "The value 144 equals the dot product of a and b in a 6-dimensional feature space. "
    "Expanding the kernel:"
)
st.latex(r"(\mathbf{a}\cdot\mathbf{b} + 1)^2 = (a_1 b_1 + a_2 b_2 + 1)^2")
st.latex(r"= a_1^2 b_1^2 + a_2^2 b_2^2 + 2a_1 a_2 b_1 b_2 + 2a_1 b_1 + 2a_2 b_2 + 1")
st.markdown("This is a dot product φ(a) · φ(b), where the feature mapping φ is:")
st.latex(r"\varphi(\mathbf{x}) = \left(x_1^2,\; x_2^2,\; \sqrt{2}\,x_1 x_2,\; \sqrt{2}\,x_1,\; \sqrt{2}\,x_2,\; 1\right)")
st.markdown(
    """
    - φ(a) = (1, 4, 2√2, √2, 2√2, 1)
    - φ(b) = (9, 16, 12√2, 3√2, 4√2, 1)
    """
)
st.latex(r"1{\cdot}9 + 4{\cdot}16 + (2\sqrt{2})(12\sqrt{2}) + (\sqrt{2})(3\sqrt{2}) + (2\sqrt{2})(4\sqrt{2}) + 1")
st.latex(r"= 9 + 64 + 48 + 6 + 16 + 1 = 144 \;\; \checkmark")
st.markdown(
    """
    <div class="key-idea">
    Both computations yield 144. Direct kernel evaluation requires one multiplication
    and one exponentiation. Explicit lifting requires constructing both 6-D vectors.
    For d = 10 on 1000-D input, the explicit space has billions of dimensions; the
    kernel computation is unchanged.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    SVM uses the full table of pairwise kernel values (the **Gram matrix**) during
    optimization. Large K(a, b) means a and b are similar in the implicit feature space.
    The parameter **C** controls strictness of fit, identical to the linear case.
    """
)

col_a, col_b = st.columns(2)
with col_a:
    poly_d = st.slider("Degree d", 2, 6, 3, 1)
with col_b:
    poly_c = st.slider("C (poly)", 0.1, 10.0, 1.0, 0.1, key="poly_c")

fig, ax = plt.subplots(figsize=(5.5, 4))
poly_model = SVC(kernel="poly", degree=poly_d, C=poly_c, coef0=1, gamma="scale").fit(X2, y2)
plot_decision_boundary(ax, poly_model, X2, y2,
                       title=f"Polynomial kernel (d={poly_d}, C={poly_c:.1f})")
style_fig(fig)
_cp1, _cp2, _cp3 = st.columns([1, 2, 1])
with _cp2:
    st.pyplot(fig, use_container_width=True)

# -------------------- RBF kernel --------------------
st.subheader("4b · RBF (Radial Basis Function) Kernel")
st.latex(r"K(\mathbf{a}, \mathbf{b}) = \exp\!\left(-\gamma\,\|\mathbf{a}-\mathbf{b}\|^2\right)")
st.markdown(
    """
    K(a, b) depends only on the Euclidean distance between a and b. The kernel value
    decays exponentially with squared distance.

    - γ small → slow decay → wide influence → smooth boundary.
    - γ large → fast decay → local influence → complex boundary (overfitting risk).

    RBF is equivalent to an infinite sum of polynomial kernels (via the Taylor series
    of exp). This corresponds to an infinite-dimensional implicit feature space.
    """
)

st.markdown("#### Worked example (γ = 0.5)")
st.markdown("Let **a = (1, 2)**, **b = (3, 4)**, **c = (1.2, 2.1)**. Compute squared distances:")
st.latex(r"\|\mathbf{a}-\mathbf{b}\|^2 = (1-3)^2 + (2-4)^2 = 4 + 4 = 8")
st.latex(r"\|\mathbf{a}-\mathbf{c}\|^2 = (1-1.2)^2 + (2-2.1)^2 = 0.04 + 0.01 = 0.05")
st.markdown("Apply the kernel with γ = 0.5:")
st.latex(r"K(\mathbf{a}, \mathbf{b}) = e^{-0.5 \times 8} = e^{-4} \approx 0.0183")
st.latex(r"K(\mathbf{a}, \mathbf{c}) = e^{-0.5 \times 0.05} = e^{-0.025} \approx 0.9753")
st.markdown(
    """
    <div class="concept-box">
    K(a, c) ≈ 0.98: a and c are close, so the kernel returns a value near 1.<br>
    K(a, b) ≈ 0.02: a and b are far apart, so the kernel returns a value near 0.<br>
    RBF outputs a similarity score in (0, 1].
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("Effect of γ on a fixed distance ‖a − b‖² = 8:")
st.latex(r"\gamma = 0.01:\; K \approx 0.923 \qquad \gamma = 0.5:\; K \approx 0.018 \qquad \gamma = 5:\; K \approx 0")
st.markdown("Larger γ produces faster decay and hence more localized decision boundaries.")

st.markdown(
    """
    The RBF kernel values populate the Gram matrix in the same way as the polynomial
    case. γ controls the boundary's locality; C controls strictness of fit. The two
    parameters are independent.
    """
)

st.markdown("**Infinite-dimensional interpretation.** The Taylor series of the exponential is:")
st.latex(r"e^{x} = 1 + x + \tfrac{x^2}{2!} + \tfrac{x^3}{3!} + \cdots")
st.markdown(
    "Substituting into the RBF formula expresses K(a, b) as an infinite sum of "
    "polynomial kernels, equivalent to a dot product in an infinite-dimensional feature "
    "space. The computation remains a single exponentiation."
)

col_a, col_b = st.columns(2)
with col_a:
    rbf_gamma = st.select_slider("γ (gamma)",
                                 options=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                                 value=0.5)
with col_b:
    rbf_c = st.slider("C (RBF)", 0.1, 10.0, 1.0, 0.1, key="rbf_c")

fig, ax = plt.subplots(figsize=(5.5, 4))
rbf_model = SVC(kernel="rbf", gamma=rbf_gamma, C=rbf_c).fit(X2, y2)
plot_decision_boundary(ax, rbf_model, X2, y2,
                       title=f"RBF kernel (γ={rbf_gamma}, C={rbf_c:.1f})")
style_fig(fig)
_cr1, _cr2, _cr3 = st.columns([1, 2, 1])
with _cr2:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    """
    <div class="key-idea">
    Low γ produces a near-linear boundary. High γ produces localized regions around
    individual fraud points — a signature of overfitting.
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Side-by-side --------------------
st.subheader("4c · Kernel Comparison")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
lin = SVC(kernel="linear", C=1).fit(X2, y2)
pol = SVC(kernel="poly", degree=3, C=1, coef0=1, gamma="scale").fit(X2, y2)
rbf = SVC(kernel="rbf", C=1, gamma="scale").fit(X2, y2)
plot_decision_boundary(axes[0], lin, X2, y2, title="Linear", show_support=False)
plot_decision_boundary(axes[1], pol, X2, y2, title="Polynomial (d=3)", show_support=False)
plot_decision_boundary(axes[2], rbf, X2, y2, title="RBF", show_support=False)
style_fig(fig)
st.pyplot(fig)

# =======================================================================
# SECTION 5: FULL MODEL EVALUATION
# =======================================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("5 · Training and Evaluation on All 30 Features")

st.markdown(
    "Training uses all 30 features with `class_weight='balanced'` to counter the class "
    "imbalance. Legitimate transactions are subsampled to keep training time under 10 seconds."
)

@st.cache_data(show_spinner=True)
def train_full_model(n_legit=6000, kernel="rbf", C=1.0, gamma="scale"):
    # Keep all fraud, subsample legit to keep training fast for the demo
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=n_legit, random_state=42)
    data = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
    X = data.drop(columns=["Class"]).values
    y = data["Class"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced", probability=False)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    y_score = model.decision_function(X_te_s)

    return {
        "model": model, "y_test": y_te, "y_pred": y_pred, "y_score": y_score,
        "n_train": len(X_tr), "n_test": len(X_te),
        "n_fraud_test": int(np.sum(y_te)),
    }

col_a, col_b, col_c = st.columns(3)
with col_a:
    full_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)
with col_b:
    full_c = st.slider("C", 0.1, 10.0, 1.0, 0.1, key="full_c")
with col_c:
    n_legit = st.selectbox("Legit samples (training speed vs realism)", [3000, 6000, 10000], index=1)

if st.button("Train the full model", type="primary"):
    with st.spinner(f"Training {full_kernel.upper()} SVM..."):
        res = train_full_model(n_legit=n_legit, kernel=full_kernel, C=full_c)

    y_te, y_pred, y_score = res["y_test"], res["y_pred"], res["y_score"]
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics row
    st.markdown("##### Test set results")
    m1, m2, m3, m4 = st.columns(4)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr, tpr, _ = roc_curve(y_te, y_score)
    roc_auc = auc(fpr, tpr)
    m1.metric("Precision (fraud)", f"{precision:.3f}")
    m2.metric("Recall (fraud)", f"{recall:.3f}")
    m3.metric("F1", f"{f1:.3f}")
    m4.metric("ROC AUC", f"{roc_auc:.3f}")

    # Confusion matrix + ROC
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Legit", "Pred Fraud"],
                yticklabels=["True Legit", "True Fraud"], ax=axes[0],
                annot_kws={"size": 9})
    axes[0].set_title("Confusion matrix", fontsize=10)

    axes[1].plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="#aaa")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC curve", fontsize=10)
    axes[1].legend(fontsize=8)
    style_fig(fig)
    _e1, _e2, _e3 = st.columns([1, 3, 1])
    with _e2:
        st.pyplot(fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="key-idea">
        Fraud detected: {tp} of {res['n_fraud_test']}. False negatives: {fn}.
        False positives: {fp}. In fraud detection the cost of a false negative
        exceeds the cost of a false positive, so recall is the primary metric.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Select kernel and C, then press 'Train the full model'.")

# =======================================================================
# SECTION 6: SVM <-> PROJECT CONNECTION
# =======================================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("6 · Mapping SVM Concepts to the Fraud Detection Task")

st.markdown(
    """
    Each SVM concept corresponds to a requirement of the fraud detection problem:

    1. **Binary classification.** The task has two classes (legit, fraud); SVM is a
       binary classifier by construction.
    2. **High-dimensional input.** The 30 features define a 30-dimensional space;
       SVM finds a separating hyperplane without dimensionality reduction.
    3. **Non-linear separability.** Fraud and legit overlap in feature space. The
       kernel trick (RBF in particular) produces non-linear decision boundaries that
       wrap around fraud clusters.
    4. **Class rarity.** The decision function depends only on support vectors —
       typically the borderline cases. Easy-to-classify legitimate points do not
       affect the boundary, which suits a highly skewed class distribution.
    5. **Asymmetric costs.** The C parameter and `class_weight='balanced'` encode the
       asymmetry between false negatives and false positives.

    ### Interpretation of the Section 5 results

    - **Recall close to 0.9** indicates the RBF boundary successfully encloses most
      fraud cases in feature space.
    - **Lower precision** reflects the cost of balanced weighting: the model accepts
      additional false positives in exchange for higher recall.
    - **ROC AUC close to 0.97** indicates strong class separation across all decision
      thresholds.

    | Concept | Role in the fraud detector |
    | --- | --- |
    | Hyperplane in ℝ³⁰ | Native handling of 30-D input |
    | Soft margin, C | Permits overlap between classes |
    | RBF kernel, γ | Non-linear boundaries around fraud clusters |
    | Support vectors | Model focuses on boundary cases |
    | `class_weight='balanced'` | Corrects for class imbalance |
    """
)

st.caption("ML course presentation — Credit Card Fraud Detection dataset (Kaggle)")
