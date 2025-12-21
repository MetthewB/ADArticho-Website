import re
import pandas as pd
from pathlib import Path

NY_DATA_PATH = "data/newyorker_caption_contest/data"


def preprocess_caption(caption):
    """
    Preprocesses a caption by:
        - Converting non-string inputs to strings or empty strings.
        - Converting text to lowercase.
        - Stripping leading/trailing whitespace.
        - Removing punctuation and special characters.
    """
    # Convert float captions to strings
    if isinstance(caption, float):
        caption = str(caption)
    # Convert non-string captions to empty strings
    elif not isinstance(caption, str):
        caption = ""
    # Convert to lowercase and remove leading/trailing whitespace
    caption = caption.lower().strip()
    # Remove punctuation and special characters
    caption = re.sub(r"[^\w\s]", "", caption)
    return caption

def filter_huge(df : pd.DataFrame, threshold) -> pd.DataFrame:
    return df[df['caption'].map(lambda s : len(s)) < threshold]

def remove_backslash_n(df : pd.DataFrame) -> pd.DataFrame:
    df['caption'] = df["caption"].map(lambda s : str.replace(s, "\n", ""))
    return df

def preprocess_caption_lr(caption, lower=True):
    """
    Preprocesses a caption by:
        - Converting non-string inputs to strings or empty strings.
        - Converting text to lowercase.
        - Stripping leading/trailing whitespace.
        - Removing punctuation and special characters.
    """
    # Convert float captions to strings
    if isinstance(caption, float):
        caption = str(caption)
    # Convert non-string captions to empty strings
    elif not isinstance(caption, str):
        caption = ""
    # Convert to lowercase and remove leading/trailing whitespace
    if lower:
        caption = caption.lower().strip()
    # Remove punctuation and special characters
    caption = re.sub(r'[^A-Za-z]+',' ', caption)
    caption = re.sub(r'\s+',' ', caption)
    return caption.strip()
    
def normalize_ny_ratings(df):

    # Columns to normalize
    column_norm = ["not_funny", "somewhat_funny", "funny"]

    for col in column_norm:
        df[col + "_norm"] = df[col] / df["votes"] if df["votes"].all() > 0 else 0.0

    return df
