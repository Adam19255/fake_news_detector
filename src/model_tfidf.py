import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def build_tfidf_logreg_model(
        max_features=50000,
        ngram_range=(1, 2),
        C=1.0,
        random_state=42
    ):
    """
    Create a TF-IDF + Logistic Regression pipeline.

    Returns:
        sklearn.Pipeline: Vectorizer + Classifier
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=5,
        sublinear_tf=True
    )

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        n_jobs=-1,
        random_state=random_state
    )

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("logreg", model)
    ])

    return pipe


def train_model(pipe, X_train, y_train):
    """
    Train the TF-IDF + Logistic Regression pipeline.
    """
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe, X_test, y_test):
    """
    Evaluate and print metrics.
    """
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_test, preds))
    return acc


def save_model(pipe, save_path="models/tfidf_logreg.joblib"):
    """
    Save pipeline (vectorizer + model together).
    """
    joblib.dump(pipe, save_path)
    print(f"Model saved to {save_path}")
