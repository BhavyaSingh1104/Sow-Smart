"""
train_model.py

Trains a rainfall-risk / crop-failure prediction model.

Model choice: XGBoost classifier.
Why: tabular data with a handful of engineered features, need feature
importance for explainability (you'll want this for your interview /
README), and it handles the class imbalance (crop_failed is ~25% of
rows) far better than plain logistic regression while still being
fast to train and easy to reason about.
"""

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

FEATURE_COLS = ["sow_week", "pre_sowing_deficit_pct", "district_avg_deficit_pct", "crop_encoded", "district_encoded"]


def main():
    df = pd.read_csv("data/features.csv")

    crop_encoder = LabelEncoder()
    district_encoder = LabelEncoder()
    df["crop_encoded"] = crop_encoder.fit_transform(df["crop"])
    df["district_encoded"] = district_encoder.fit_transform(df["district"])

    X = df[FEATURE_COLS]
    y = df["crop_failed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handle imbalance
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Evaluation on held-out test set ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["survived", "failed"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("=== Feature importances (what drives the risk score) ===")
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:28s} {imp:.3f}")

    joblib.dump(model, "models/risk_model.joblib")
    joblib.dump(crop_encoder, "models/crop_encoder.joblib")
    joblib.dump(district_encoder, "models/district_encoder.joblib")
    print("\nSaved: models/risk_model.joblib, crop_encoder.joblib, district_encoder.joblib")


if __name__ == "__main__":
    main()
