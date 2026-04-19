# SVM Credit Card Fraud — 30-Minute Demo Script

Two presenters:
- **Teammate (T)** — Sections 1, 2, 3 (first ~13 min)
- **You (Y)** — Sections 4, 5, 6 (next ~13 min) + Q&A

Target: 30 min total. Opener 1 min + core 26 min + Q&A 3 min.

---

## Opener (1 min) — both presenters on stage

> **Y:** "Hey everyone. Today we're demoing **Support Vector Machines**, and instead of
> showing it on a toy dataset, we're going to use a real problem — **credit card fraud
> detection**. The whole demo is an interactive app; every concept has a slider you can
> play with, so you'll see the model react in real time."
>
> **T:** "I'll cover the foundation — the problem, how SVM draws a line, and the soft
> margin idea. Then [Your name] will take over with the kernel trick, the actual model,
> and why SVM is the right choice for fraud. Let's go."

*(Open localhost:8501. Scroll to the top.)*

---

## SECTION 1 — THE PROBLEM (2 min) · T

*(Scroll to Section 1. Point at the bar chart.)*

> "Credit card fraud is a **binary classification** problem — legit or fraud. Two things
> make it uniquely hard:
>
> - **Imbalance:** fewer than 1 in 500 transactions are fraud — look at the log-scale
>   chart, legit is 500× taller than fraud.
> - **Overlap:** fraud doesn't sit in its own corner of feature space; it hides among
>   normal transactions.
>
> Why this matters: a model that just predicts 'never fraud' already hits **99.83%
> accuracy** — and catches zero fraudsters. So **accuracy is the wrong metric here** —
> we'll come back to that."

---

## SECTION 2 — A LINE BETWEEN TWO GROUPS (5 min) · T

*(Scroll down.)*

> "Let's start with the core idea. SVM looks for a line — or in higher dimensions, a
> **hyperplane** — that separates two classes with the **widest possible margin**."

*(Point to apples-vs-oranges plot.)*

> "Forget fraud for a second. Here are apples (red) and oranges (orange) by weight and
> diameter. SVM draws the **solid line** between them. The **dashed lines** mark the
> margin — the widest empty strip possible without touching any point. Those **circled
> points** on the dashed lines are the **support vectors**. That's the key property:
> *delete any other point and the line doesn't move* — only support vectors define it."

*(Scroll to credit card 2D plot.)*

> "Now the real data: V14 and V17, two of the most predictive fraud features. Same idea,
> messier picture. Red dots are fraud, teal is legit. Notice there's already some
> overlap — that'll be important in a minute."

*(Scroll to "How does SVM find this line?")*

> "Mathematically, any line is **w · x + b = 0**. **w** is perpendicular to the line, **b**
> shifts it. SVM finds the w and b that **maximize the margin**, which turns out to be
> 2 over the length of w. So minimizing ‖w‖ gives the widest margin — that's the
> optimization we solve."

*(Deep breath. Transition.)*

> "But this only works if the data is cleanly separable. Real fraud data isn't. That's
> why we need…"

---

## SECTION 3 — HARD VS SOFT MARGIN (5 min) · T

*(Scroll to Section 3.)*

> "**Hard margin** says 'no point is allowed to violate the margin.' Only works if
> perfectly separable. **Soft margin** allows a few violations for a wider, more stable
> boundary."

*(Show the formula.)*

> "Here's the SVM objective. Two terms:
> - **½ ‖w‖²** — minimize this to make the margin wide.
> - **C × Σ ξᵢ** — penalize the total slack.
>
> **ξᵢ (slack)** is how much point *i* violates the margin. Zero = fine, between zero
> and one = inside the margin but on the correct side, greater than one = misclassified.
> **C** controls the tradeoff: big C punishes violations hard (overfits); small C
> tolerates them (generalizes)."

*(Point to cross-validation explanation.)*

> "How do we pick C? Not from training scores — that always prefers big C. We use
> **k-fold cross-validation**: split the training data into 5 chunks, train on 4, test on
> 1, rotate. Average the 5 scores. Pick the C with the best average. Same procedure picks
> gamma or degree later."

*(Drag the C slider.)*

> "Watch what happens. At C equals 0.01, the margin is super wide, lots of points inside
> — mostly smoothed out. At C equals 50, the margin is narrow and the model tries hard to
> get every training point right. The support-vector count changes too — lower C means
> more support vectors."

*(Hand off.)*

> **T:** "That's the foundation. [Your name], take it away."

---

## SECTION 4 — THE KERNEL TRICK (7 min) · Y

*(Scroll to Section 4.)*

> "OK — what if no straight line works? Classic example: a drug that cures at **medium**
> doses but fails at both **low** and **high** doses. No 1D threshold separates them."

*(Point at 1D/2D dosage plot.)*

> "But if I add a second axis — **dosage squared** — suddenly the middle curves above the
> extremes, and a flat line splits them cleanly. That's the SVM insight: **lift the data
> into a higher dimension where a line works again.**
>
> The problem: actually lifting every point into a high dimension is expensive. That's
> where the **kernel trick** comes in — we compute the *dot product* as if we lifted,
> without actually lifting."

### 4a — Polynomial kernel

*(Show K(a, b) = (a·b + r)^d.)*

> "Let me show it concretely. Two points **a = (1, 2)** and **b = (3, 4)**, degree 2, r=1.
>
> - Dot product: (1)(3) + (2)(4) = 11.
> - Plug in: (11 + 1)² = **144**.
>
> That single number **144 is the dot product the two points *would* have after being
> lifted into a 6-dimensional space**. The expansion below shows it: I can lift both
> points into a specific 6-D phi, take their dot product explicitly, and get 144. But
> computing K directly is one multiply and one power — way cheaper."

*(Drag the polynomial degree slider, show boundary curving.)*

> "Slide degree from 2 to 6 — watch the boundary get more flexible. C works the same as
> before — strictness of fit."

### 4b — RBF kernel (5 min)

*(Scroll down.)*

> "Now the **RBF kernel** — the real workhorse. The formula uses squared distance and
> gamma.
>
> Same a and b, plus c equals (1.2, 2.1) — close to a.
> - Distance a to b = 8, distance a to c = 0.05.
> - K(a, b) = e^(-4) ≈ **0.02** (far points → low similarity).
> - K(a, c) = e^(-0.025) ≈ **0.98** (close → high similarity).
>
> **RBF is a similarity score from 0 to 1.** Close = 1, far = 0. It behaves like a
> weighted nearest-neighbour classifier."

*(Show gamma sweep: 0.01, 0.5, 5.)*

> "Gamma controls how fast similarity drops off. Small gamma = faraway points still
> matter = smooth boundary. Large gamma = only very close points matter = tight, wavy
> boundary."

*(Drag the gamma slider.)*

> "Watch the boundary. At 0.01, it's almost a straight line. At 10, it's making islands
> around individual fraud points — that's classic overfitting."

*(Point at Taylor expansion.)*

> "Why is RBF special? If you expand the exponential as a Taylor series, you get an
> **infinite sum of polynomial kernels of every degree**. That means RBF secretly
> operates in an **infinite-dimensional space**, but computes with a single exp call."

*(Scroll to side-by-side.)*

> "Here's the three kernels on the same fraud data: linear can't wrap the clusters,
> polynomial curves a bit, RBF wraps them tightly. That's why RBF is the default
> choice for most problems."

---

## SECTION 5 — FULL MODEL (4 min) · Y

*(Scroll to Section 5.)*

> "Time for the real thing. All 30 features now, not just two. Kernel: RBF. C equals 1.
> `class_weight='balanced'` — which tells SVM to weight fraud cases more heavily so it
> doesn't just ignore them."

*(Click **Train the full model**.)*

> "Takes about 5 seconds… [wait] …and done. Here's what we got:
>
> - **Precision** — of everything flagged as fraud, what fraction actually was.
> - **Recall** — of all real fraud, what fraction we caught.
> - **F1** — harmonic mean of those two.
> - **ROC AUC** — overall separation quality.
>
> Our model catches [**X**] out of [**Y**] real fraud cases with AUC [**Z**]. Remember
> how a 'never fraud' model hits 99.83% accuracy? Look at our recall — *that's* the
> meaningful number for fraud detection."

---

## SECTION 6 — WHY SVM FOR FRAUD (2 min) · Y

*(Scroll to Section 6.)*

> "To close, why did we pick SVM for this problem? Five reasons:
>
> 1. Binary classification — SVM's native shape.
> 2. 30 features → hyperplane in 30-D, no manual work.
> 3. Fraud and legit overlap → **kernel trick**, RBF wraps tight boundaries around fraud
>    clusters.
> 4. Fraud is rare → **support vectors** focus on borderline cases, ignore easy points.
>    Perfect for minority classes.
> 5. Missing a fraud costs more than a false alarm → **C** and `class_weight='balanced'`
>    bake that cost in.
>
> Every piece of SVM theory we showed maps to one design decision in this detector.
> That's the full loop — the math isn't abstract, it's *why the model works on fraud*."

---

## Q&A (3 min)

Likely questions + prepped answers:

**Q: Why not just use logistic regression / random forest?**
> "Logistic regression gives a linear boundary — struggles with overlapping clusters.
> Random forests handle it fine and are often faster; SVM wins when the decision
> boundary is genuinely curved and the dataset is moderate-sized. For pure performance
> on fraud at scale, ensembles often win — we picked SVM because the *concepts* (margin,
> kernel) are more educational."

**Q: How long does it take to train on the full dataset?**
> "SVM is O(n²) to O(n³) in training size. On the full 284K rows, RBF SVM is slow —
> hours. That's why we subsample legit transactions. Production systems often use a
> linear SVM or gradient boosting instead."

**Q: Can't you just oversample the fraud class?**
> "Yes — SMOTE or random oversampling are standard. `class_weight='balanced'` is the
> lightweight version of the same idea. We kept it simple for the demo."

**Q: Why the PCA-transformed features V1–V28?**
> "The original features are confidential — real credit card data. Kaggle released it
> after PCA so no one can reverse-engineer cardholders."

**Q: Does SVM work on a GPU?**
> "sklearn's SVM is CPU-only. GPU SVM exists via `cuML` (RAPIDS) but needs specific
> CUDA setup. For our data size, CPU is fine."

**Q: What's the difference between C and gamma?**
> "C controls *strictness of fit* — how much we punish training mistakes. Gamma
> controls *shape of the RBF boundary* — local vs smooth. They're independent knobs."

---

## Speaker cues

- **Pauses:** after every worked example, pause 2 seconds for the audience to absorb.
- **Audience engagement:** after the C slider in Section 3 and the gamma slider in
  Section 4, ask: "What do you think happens if I crank this to the max?" Then drag it.
- **Time checkpoint:** At the end of Section 3 (T's handoff), you should be at ~13 min.
  If behind, skip the "two rules" explanation in Section 2 and move on.
- **Backup:** If the app breaks, the `svm_fraud_model.py` script runs standalone —
  `python svm_fraud_model.py` produces the same output in the terminal + a PNG.
