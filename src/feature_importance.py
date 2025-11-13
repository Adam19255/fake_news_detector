import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_top_features(pipe, top_n=30):
    """
    Extract top positive and negative features from TF-IDF + Logistic Regression pipeline.
    
    Returns:
        pd.DataFrame with words and their coefficients.
    """
    vectorizer = pipe.named_steps["tfidf"]
    model = pipe.named_steps["logreg"]

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]  # binary classification → one row

    coef_df = pd.DataFrame({
        "word": feature_names,
        "coef": coefs
    })

    # Negative = class 0 (fake)
    # Positive = class 1 (real)
    fake_top = coef_df.sort_values("coef").head(top_n)
    real_top = coef_df.sort_values("coef", ascending=False).head(top_n)

    return fake_top, real_top


def plot_top_features(fake_top, real_top, top_n=30):
    """
    Plot bar charts for fake vs real top words.
    """
    plt.figure(figsize=(14, 7))

    # Fake-news pushing words
    plt.subplot(1, 2, 1)
    plt.barh(fake_top["word"], fake_top["coef"], color="red")
    plt.title(f"Top {top_n} Fake-News Indicators (negative coefficients)")
    plt.xlabel("Coefficient")
    plt.gca().invert_yaxis()

    # Real-news pushing words
    plt.subplot(1, 2, 2)
    plt.barh(real_top["word"], real_top["coef"], color="green")
    plt.title(f"Top {top_n} Real-News Indicators (positive coefficients)")
    plt.xlabel("Coefficient")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
