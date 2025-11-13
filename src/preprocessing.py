import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Basic text cleaning for TF-IDF model.
    Note: We do NOT remove stopwords yet. We'll experiment later.

    Steps:
    - Lowercase
    - Remove URLs
    - Remove HTML tags
    - Remove non-alphanumeric chars except basic punctuation
    - Collapse extra spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"<.*?>", " ", text)            # HTML tags
    text = re.sub(r"[^a-z0-9\s\.,;:!?']", " ", text)  # Keep basic punctuation

    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the 'content' field.
    Creates a new field 'clean_content'.
    """
    df["clean_content"] = df["content"].apply(clean_text)
    return df


def split_dataset(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split dataset into train, validation, and test sets.

    Process:
    - First split into train + temp (test+val)
    - Then split temp into validation and test
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        stratify=df["label"],
        random_state=random_state
    )

    val_ratio = val_size / (test_size + val_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_ratio,
        stratify=temp_df["label"],
        random_state=random_state
    )

    return train_df, val_df, test_df
