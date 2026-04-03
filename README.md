# 🤖 Machine Learning Workflow in Python

> *A structured, end-to-end walkthrough of the modern ML pipeline — from raw data to evaluated models — using scikit-learn, pandas, and Python best practices.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20toolkit-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What Is This?

Most ML tutorials teach algorithms in isolation. This repository takes a different approach: it teaches the **complete ML workflow** as a repeatable, professional process — the way it's actually done in production data science.

Covering everything from exploratory analysis and feature engineering through model selection, evaluation, and interview-grade problem solving, this is a practical reference for anyone moving from "I know the theory" to "I can ship a model."

---

## 📁 Repository Structure

```
MachineLearningWorkFlow/
│
├── WorkFlow_1.ipynb          ← Core ML workflow: Part 1
├── Workflow_1_1.ipynb        ← Core ML workflow: Part 1 (extended)
├── Workflow_2.ipynb          ← Core ML workflow: Part 2
│
├── session1.ipynb            ← Hands-on session: fundamentals
├── session2.ipynb            ← Hands-on session: advanced topics
│
├── scikit-learn/             ← Focused scikit-learn deep dives
│
├── LP DS Coding Interview/   ← Data science coding interview prep
│
└── data/                     ← Datasets used across notebooks
```

---

## 🗺️ The ML Workflow Covered

Each stage of the pipeline is treated as a first-class concern — not just an afterthought before model fitting:

```
 1. Problem Definition
       │   What are we predicting? Classification or regression?
       │   What does success look like?
       ▼
 2. Data Loading & Exploration (EDA)
       │   Shape, types, distributions
       │   Missing values, outliers, correlations
       │   Target variable analysis
       ▼
 3. Feature Engineering & Preprocessing
       │   Encoding categoricals (OneHotEncoder, OrdinalEncoder)
       │   Scaling numerics (StandardScaler, MinMaxScaler)
       │   Imputation strategies
       │   Pipeline construction
       ▼
 4. Model Selection
       │   Baselines first (DummyClassifier/Regressor)
       │   Linear models → tree-based → ensembles
       │   scikit-learn API: fit / predict / score
       ▼
 5. Model Evaluation
       │   Cross-validation (KFold, StratifiedKFold)
       │   Metrics: accuracy, F1, ROC-AUC, RMSE, R²
       │   Confusion matrices, learning curves
       ▼
 6. Hyperparameter Tuning
       │   GridSearchCV, RandomizedSearchCV
       │   Avoiding data leakage inside pipelines
       ▼
 7. Interpretation & Communication
             Feature importances, SHAP values
             Communicating results to non-technical stakeholders
```

---

## 🧰 scikit-learn Toolkit

The `scikit-learn/` folder provides focused notebooks on the library's key components:

| Component | Purpose |
|-----------|---------|
| `Pipeline` | Chain preprocessing + model into a single estimator |
| `ColumnTransformer` | Apply different transforms to different feature types |
| `cross_val_score` | Robust out-of-sample evaluation |
| `GridSearchCV` | Exhaustive hyperparameter search |
| `RandomizedSearchCV` | Efficient hyperparameter sampling |
| Classification models | LogisticRegression, RandomForest, GradientBoosting, SVM |
| Regression models | LinearRegression, Ridge, Lasso, ElasticNet |
| Clustering | KMeans, DBSCAN |

---

## 💼 Data Science Coding Interview Prep

The `LP DS Coding Interview/` folder contains curated challenges covering the most commonly tested topics in data science interviews:

- **Python & pandas**: data wrangling under time pressure
- **SQL-style operations** with pandas
- **Statistical reasoning**: distributions, hypothesis testing, A/B test design
- **ML model design questions**: feature selection, overfitting, imbalanced classes
- **Complexity and optimization** of common DS operations

---

## ⚡ Quick-Start Example

A complete scikit-learn workflow in under 30 lines:

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('data/dataset.csv')
X, y = df.drop('target', axis=1), df['target']

# Define feature types
num_features = X.select_dtypes(include='number').columns.tolist()
cat_features = X.select_dtypes(include='object').columns.tolist()

# Build preprocessing + model pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Evaluate with cross-validation — no data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Final evaluation on held-out test set
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

---

## 🔑 Key Principles Throughout

**Use Pipelines — always.** Fitting a scaler on training data and applying it to test data separately is a data leakage bug. Pipelines prevent this and make deployment trivial.

**Baseline first.** A `DummyClassifier` that always predicts the majority class is your benchmark — if your model barely beats it, something is wrong.

**Cross-validate, don't just split.** A single train/test split is a noisy estimate of generalisation. `cross_val_score` with 5 or 10 folds gives a much more reliable picture.

**Metric choice matters.** Accuracy is misleading on imbalanced datasets. Match your metric to your business objective: F1 for imbalanced classification, ROC-AUC for ranking, RMSE for regression with large-error penalties.

---

## 🛠️ Installation

```bash
git clone https://github.com/alketcecaj12/MachineLearningWorkFlow.git
cd MachineLearningWorkFlow

pip install scikit-learn pandas numpy matplotlib seaborn jupyter

jupyter notebook
```

---

## 🗺️ Suggested Learning Path

```
session1.ipynb              ← Start here: foundations and EDA
      ↓
WorkFlow_1.ipynb            ← Full pipeline: data → model → evaluation
      ↓
Workflow_1_1.ipynb          ← Deeper dive: feature engineering & encoding
      ↓
Workflow_2.ipynb            ← Advanced: ensembles, tuning, interpretation
      ↓
session2.ipynb              ← Applied: real-world dataset end-to-end
      ↓
scikit-learn/               ← API deep dives: Pipelines, GridSearch, etc.
      ↓
LP DS Coding Interview/     ← Test yourself: interview-grade challenges
```

---

## 👤 Author

**Alket Cecaj**
Quantitative Risk Analyst & Data Scientist | PhD | Copenhagen
12+ years in data science · Credit Risk Modelling · ML in Production
📎 [GitHub @alketcecaj12](https://github.com/alketcecaj12)

---

## ⭐ If this repo helped you think about ML as a *process* rather than just algorithms — give it a star!
