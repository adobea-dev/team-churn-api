"""Train and save the churn pipeline (UCI Iranian Churn, id=563).

Run from the repo root:
    uv run python ml_platform/training/churn/train.py
"""

import json
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

ARTIFACT_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    dataset = fetch_ucirepo(id=563)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0]

    # Collapse stray double spaces in column names so they match the API mapping
    X.columns = [" ".join(c.split()) for c in X.columns]

    print("Shape:", X.shape)
    print("Features:", X.columns.tolist())
    print("Target distribution:\n", y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, predictions, output_dict=True)
    auc = roc_auc_score(y_test, probabilities)

    print(classification_report(y_test, predictions))
    print("ROC-AUC:", round(auc, 4))
    print("Sample probabilities:", probabilities[:5].round(3))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_DIR / "churn_pipeline.joblib")
    (ARTIFACT_DIR / "feature_names.json").write_text(
        json.dumps(X.columns.tolist(), indent=2)
    )
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "roc_auc": round(auc, 4),
                "churn_precision": round(report["1"]["precision"], 4),
                "churn_recall": round(report["1"]["recall"], 4),
                "churn_f1": round(report["1"]["f1-score"], 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
            },
            indent=2,
        )
    )
    print(f"Saved artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
