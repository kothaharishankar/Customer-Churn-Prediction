"""Train several churn models, compare them, and save the best one.

Run from the project root:
    python -m app.services.train_model --data-path data/train.csv --output app/models/churn_pipeline.pkl

Models compared: Logistic Regression, Decision Tree, Random Forest and (if installed) XGBoost.
The winner is chosen by 5-fold cross-validated ROC-AUC on the training data, then scored once on the
held-out test set (data/test.csv if it exists, otherwise a 20% split).
"""
import argparse
import json
import os
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.services.preprocessing import CATEGORICAL, ensure_column_order, split_X_y

try:  # optional: python -m pip install xgboost
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

SEED = 42


def build_pipeline(clf, numeric_cols):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def candidate_models(pos_weight):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=SEED),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.9, colsample_bytree=0.9, scale_pos_weight=pos_weight,
                                          eval_metric="logloss", random_state=SEED, n_jobs=1)
    else:
        print("(XGBoost not installed - skipping. Install with: python -m pip install xgboost)")
    return models


def evaluate(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
    }


def main(data_path: str, output: str):
    warnings.filterwarnings("ignore")
    df = pd.read_csv(data_path)
    X, y = split_X_y(df.copy())
    X = ensure_column_order(X)

    test_path = os.path.join(os.path.dirname(data_path), "test.csv")
    if os.path.abspath(data_path) != os.path.abspath(test_path) and os.path.exists(test_path) and len(df) > 500:
        X_tr, y_tr = X, y
        X_te, y_te = split_X_y(pd.read_csv(test_path))
        X_te = ensure_column_order(X_te)
        print(f"Train rows: {len(X_tr)} | test rows (data/test.csv): {len(X_te)}")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        print(f"Train rows: {len(X_tr)} | test rows (20% split): {len(X_te)}")

    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, clf in candidate_models(pos_weight).items():
        pipe = build_pipeline(clf, numeric_cols)
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_tr, y_tr)
        results[name] = {"cv_roc_auc": round(float(cv_auc), 4), **evaluate(y_te, pipe.predict_proba(X_te)[:, 1])}
        fitted[name] = pipe

    best = max(results, key=lambda n: results[n]["cv_roc_auc"])

    print(f"\n{'Model':22s}{'CV AUC':>8s}{'Test AUC':>10s}{'Accuracy':>10s}{'Precision':>10s}{'Recall':>8s}{'F1':>7s}")
    for n, r in results.items():
        mark = "  <-- selected" if n == best else ""
        print(f"{n:22s}{r['cv_roc_auc']:8.3f}{r['roc_auc']:10.3f}{r['accuracy']:10.3f}"
              f"{r['precision']:10.3f}{r['recall']:8.3f}{r['f1']:7.3f}{mark}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    joblib.dump(fitted[best], output)
    comparison = os.path.join(os.path.dirname(output) or ".", "model_comparison.json")
    with open(comparison, "w") as f:
        json.dump({"selected": best, "results": results}, f, indent=2)
    print(f"\nSaved pipeline ({best}) to {output}")
    print(f"Saved comparison table to {comparison}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output", default="app/models/churn_pipeline.pkl")
    args = ap.parse_args()
    main(args.data_path, args.output)