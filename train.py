"""Train the model and save it for the API.

Uses the same final model as the model-selection task: StandardScaler + Logistic Regression (C=1)
on the Breast Cancer Wisconsin (Diagnostic) dataset. Positive class (1) = malignant.
Run:  python train.py        (the Dockerfile runs this during the image build)
"""
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).resolve().parent / "model"


def main():
    data = load_breast_cancer()
    X = data.data
    y = (data.target == 0).astype(int)   # scikit-learn codes malignant as 0; flip so malignant = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=1, max_iter=2000))])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
               "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
               "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    meta = {"model": "StandardScaler + Logistic Regression (C=1)",
            "task": "Binary classification: malignant (1) vs benign (0)",
            "dataset": "Breast Cancer Wisconsin (Diagnostic), scikit-learn built-in",
            "features": list(data.feature_names),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "scikit_learn_version": sklearn.__version__}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved model to", MODEL_DIR)
    print("Test metrics:", meta["test_metrics"])


if __name__ == "__main__":
    main()
