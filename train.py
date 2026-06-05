import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- REPLACE THIS SECTION SATURDAY WITH ACTUAL DATA ----
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
FEATURE_NAMES = list(X.columns)
NUM_FEATURES = len(FEATURE_NAMES)
# --------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("willovate_ml")

with mlflow.start_run():
    # Log params
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("num_features", NUM_FEATURES)
    mlflow.log_param("train_samples", len(X_train))

    # Train
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=n_estimators, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Log model to MLflow
    mlflow.sklearn.log_model(pipeline, "model")

    # Also save as pickle for the API
    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # Save training data stats for drift detection later
    X_train.describe().to_csv("train_stats.csv")
    mlflow.log_artifact("train_stats.csv")

    print("Model saved to model.pkl")
    print("MLflow run complete. Run `mlflow ui` to view dashboard.")