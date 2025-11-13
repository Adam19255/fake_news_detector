import pandas as pd
import os


def load_fake_real_dataset(data_dir: str = "data"):
    """
    Load the Kaggle Fake and Real News dataset.

    Args:
        data_dir (str): Path to the 'data' folder containing Fake.csv and True.csv.

    Returns:
        pd.DataFrame: Combined dataset with columns ["title", "text", "label"].
                      label = 0 -> fake, 1 -> real
    """
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            f"Dataset files not found inside {data_dir}. "
            "Expected Fake.csv and True.csv."
        )

    # Read datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels
    fake_df["label"] = 0   # fake
    true_df["label"] = 1   # real

    # Combine
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Clean missing values
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Combine title + body into one field
    df["content"] = df["title"] + " " + df["text"]

    return df


def basic_dataset_info(df: pd.DataFrame):
    """
    Print basic dataset statistics for EDA.
    """
    print("\n=== DATASET INFO ===")
    print(df.info())

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== CLASS DISTRIBUTION ===")
    print(df["label"].value_counts())

    print("\n=== AVERAGE TEXT LENGTH (WORDS) ===")
    df["word_count"] = df["content"].apply(lambda x: len(x.split()))
    print(df["word_count"].describe())
