
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation.json"
REPORT_PATH = PROCESSED_DIR / "classification_report.txt"

CLASSES = ["negative", "neutral", "positive"]


def load_dataset(filename):
    """Load text and sentiment labels from a CSV file."""
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    required_columns = {"text", "sentiment"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {sorted(missing)}"
        )

    df = df[["text", "sentiment"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["sentiment"] = (
        df["sentiment"].fillna("").astype(str).str.strip().str.lower()
    )

    # Remove blank texts and invalid labels
    df = df[
        (df["text"] != "")
        & (df["sentiment"].isin(CLASSES))
    ].copy()

    if df.empty:
        raise ValueError(f"No valid records found in {filename}")

    return df


def evaluate_model(model, dataset, dataset_name):
    """Calculate classification metrics."""
    actual = dataset["sentiment"]
    predicted = model.predict(dataset["text"])

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(actual, predicted)

    print(f"\n{'=' * 55}")
    print(f"{dataset_name} EVALUATION")
    print(f"{'=' * 55}")
    print(f"Records evaluated : {len(dataset):,}")
    print(f"Accuracy          : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro precision   : {precision:.4f}")
    print(f"Macro recall      : {recall:.4f}")
    print(f"Macro F1-score    : {f1:.4f}")

    report = classification_report(
        actual,
        predicted,
        labels=CLASSES,
        target_names=CLASSES,
        zero_division=0,
        digits=4,
    )
    print("\nClassification report:")
    print(report)

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=CLASSES,
    )

    print("Confusion matrix (actual rows, predicted columns):")
    print("Labels:", CLASSES)
    print(matrix)

    return {
        "dataset": dataset_name,
        "records_evaluated": int(len(dataset)),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "class_labels": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=CLASSES,
            target_names=CLASSES,
            zero_division=0,
            output_dict=True,
        ),
    }, report


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sentiment datasets...")
    train_df = load_dataset("train_df.csv")
    val_df = load_dataset("val_df.csv")
    test_df = load_dataset("test_df.csv")

    print(f"Training records  : {len(train_df):,}")
    print(f"Validation records: {len(val_df):,}")
    print(f"Test records      : {len(test_df):,}")

    print("\nTraining class distribution:")
    print(train_df["sentiment"].value_counts())

    print("\nBuilding TF-IDF + Logistic Regression pipeline...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining started. This may take a few minutes...")
    start_time = time.time()

    model.fit(train_df["text"], train_df["sentiment"])

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds.")

    print("\nSaving trained model...")
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Model saved to: {MODEL_PATH}")

    # Evaluate on validation data and held-out test data
    validation_metrics, _ = evaluate_model(
        model, val_df, "VALIDATION"
    )
    test_metrics, test_report = evaluate_model(
        model, test_df, "TEST"
    )

    results = {
        "model": "TF-IDF + Logistic Regression",
        "training_records": int(len(train_df)),
        "validation": validation_metrics,
        "test": test_metrics,
        "training_seconds": round(elapsed, 2),
        "features": {
            "ngram_range": [1, 2],
            "max_features": 50000,
            "class_weight": "balanced",
            "classes": CLASSES,
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("SENTIMENT CLASSIFICATION — TEST REPORT\n\n")
        file.write(test_report)
        file.write("\n\nConfusion matrix (actual rows, predicted columns):\n")
        file.write(str(test_metrics["confusion_matrix"]))
        file.write("\n")

    print("\n" + "=" * 55)
    print("TRAINING PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 55)
    print(f"Trained model : {MODEL_PATH}")
    print(f"Metrics JSON  : {METRICS_PATH}")
    print(f"Test report   : {REPORT_PATH}")
    print("\nNext step: connect this saved model to app.py.")


if __name__ == "__main__":
    main()
