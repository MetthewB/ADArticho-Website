import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import os
from src.data.preprocess import normalize_ny_ratings
from datetime import datetime, timedelta


class NewYorkerDataset(Dataset):
    """
    A dataset for the New Yorker data.
    Args:
        - data_path: Path to the data folder of the NewYorker dataset
        - caption_list: List of images to include in the dataset. If empty all images are added.
    Implements:
        - __len__: Returns the number of samples in the dataset.
        - __getitem__: Returns a sample from the dataset at the given index.
    """

    def __init__(self, data_path, caption_list=None):
        super().__init__()
        self.data_path = Path(
            data_path
        )  # Ensure data_path is a Path object for compatibility with pathlib
        self.data = self._load_data(self.data_path, caption_list)
        self.normalized_data = normalize_ny_ratings(self.data)

    def _load_data(self, data_path, caption_list):
        dfs = []

        # If caption_list is missing add all existing images to the dataset
        if (caption_list is None) or (len(caption_list) == 0):
            caption_list = [
                int(f.replace(".csv", ""))
                for f in os.listdir(data_path)
                if f.endswith(".csv")
            ]

        for img_nbr in caption_list:
            file_path = data_path / f"{img_nbr}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Drop  missing captions
                df = df[pd.notna(df["caption"])]

                df["votes"] = df["funny"] + df["somewhat_funny"] + df["not_funny"]
                df.loc[df["votes"] > 0, "mean"] = (
                    3 * df["funny"] + 2 * df["somewhat_funny"] + df["not_funny"]
                ) / df["votes"]
                df.loc[df["votes"] == 0, "mean"] = 0
                # Sort by mean and add a 'rank' column if it doesn't exist
                df.sort_values(["mean"], inplace=True, ascending=False)

                df["rank"] = [i for i in range(len(df))]
                # Add an 'image_id' column for reference
                df["image_id"] = int(img_nbr)
                dfs.append(df)
        # Combine all dataframes into a single dataframe
        return (
            pd.concat(dfs, ignore_index=True)
            .sort_values(["image_id", "rank"])
            .reset_index(drop=True)
        )

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return a specific row (sample) from the dataset
        return self.data.iloc[idx]


class OxfordDataset(Dataset):
    """
    A dataset for the Oxford data.
    Implements:
        - __len__: Returns the number of samples in the dataset.
        - __getitem__: Returns a sample from the dataset at the given index.
    """

    def __init__(self, data_path):
        super().__init__()
        self.data = self._load_data(data_path + "/oxford_hic_data.csv")

    def _load_data(self, data_path):
        # Load the dataset, remove null image ids captions, and preprocess the 'funny_score' column
        df = pd.read_csv(
            data_path,
            dtype={"image_id": str, "caption": str},
            low_memory=False,
            quotechar='"',
        )
        df = df[pd.notna(df["image_id"])]
        # Drop  missing captions
        df = df[pd.notna(df["caption"])]
        df["funny_score"] = df["funny_score"].str.replace(",", "", regex=True)
        df["funny_score"] = pd.to_numeric(df["funny_score"], errors="coerce")
        # Sort by 'image_id' and 'funny_score' in descending order
        df = df.sort_values(
            by=["image_id", "funny_score"], ascending=[True, False]
        ).reset_index(drop=True)
        # Assign ranks within each 'image_id' group
        df["rank"] = df.groupby("image_id").cumcount().astype(int)
        # Filter out groups where all 'funny_score' values are 0
        df = df.groupby("image_id").filter(
            lambda group: group["funny_score"].max() > 0.0
        )
        return df.reset_index(drop=True)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return a specific row (sample) from the dataset
        return self.data.iloc[idx]


def load_classified_dataset(dataset_path, low_quantile=0, dates_col=False):
    """
    Loads the dataset with the columns: Topic, matching_score and date for Newyorker
    Ards: 
        dataset_path (str): Path to the dataset
        low_quantile (float): Quantile containing the lower matching_score for each topic
        dates_col: True for adding a date column for New Yorker dataset
    
    Return:
        dataset_classified (pd.Dataframe): Dataframe containing the captions, their topic and dates
    """
    # Read file and remove extra index column
    dataset_classified = pd.read_csv(dataset_path).drop(columns="Unnamed: 0")

    # Classify captions with matching_Score lower than threshold as 'Other'
    def reclassify_low_quantile(group):
        # Do not reclassify existing "Other"
        if group.name == "Other":
            return group

        threshold = group["matching_score"].quantile(low_quantile)
        group.loc[group["matching_score"] < threshold, "Topic"] = "Other"
        return group

    dataset_classified = dataset_classified.groupby("Topic", group_keys=False).apply(
        reclassify_low_quantile
    )

    if dates_col:
        # Create date dictionnary
        ## Dictionnary of dates of publication of winner. Some dates are accurate and some were estimated.
        img_dates = {
            512: "2016-03-21",
            513: "2016-03-27",
            514: "2016-04-03",
            515: "2016-04-10",
            586: "2017-10-16",
            615: "2018-05-28",
            819: "2022-10-03",
            818: "2022-09-26",
            817: "2022-09-19",
            816: "2022-09-12",
            815: "2022-09-05",
            814: "2022-08-29",
            813: "2022-08-22",
            812: "2022-08-15",
            811: "2022-08-08",
            810: "2022-08-01",
            826: "2022-11-21",
            825: "2022-11-14",
            824: "2022-11-07",
            823: "2022-10-31",
            822: "2022-10-24",
            821: "2022-10-17",
            829: "2022-12-12",
            828: "2022-12-12",
            830: "2022-12-19",
            831: "2023-01-02",
        }

        # Convert dates to datetime format
        for k in img_dates:
            img_dates[k] = datetime.strptime(img_dates[k], "%Y-%m-%d")
        # Sort keys
        keys = sorted(img_dates.keys())
        full_dates = {}

        # ---- Fill backwards to 510 ----
        first_idx = keys[0]
        first_date = img_dates[first_idx]

        for k in range(510, first_idx):
            full_dates[k] = first_date - timedelta(weeks=(first_idx - k))

        # ---- Fill intervals ----
        for i in range(len(keys) - 1):
            start_idx = keys[i]
            end_idx = keys[i + 1]
            start_date = img_dates[start_idx]

            for j in range(start_idx, end_idx):
                full_dates[j] = start_date + timedelta(weeks=j - start_idx)

        # ---- Continue to 895 ----
        last_idx = keys[-1]
        last_date = img_dates[last_idx]

        for k in range(last_idx, 896):
            full_dates[k] = last_date + timedelta(weeks=k - last_idx)

        # Convert format
        dates_dict = {k: full_dates[k].strftime("%Y.%m.%d") for k in sorted(full_dates)}

        # Add date column to Newyorker dataset
        dataset_classified["date"] = (
            dataset_classified["image_id"]
            .map(dates_dict)
            .pipe(pd.to_datetime, format="%Y.%m.%d")
        )

        # Add year column
        dataset_classified["year"] = dataset_classified["date"].dt.year
    return dataset_classified


class MemotionDataset(Dataset):
    """
    Dataset for the Memotion CSV in the format:

    id,image_name,text_ocr,text_corrected,humour,sarcasm,offensive,motivational,overall_sentiment

    - Uses `text_corrected` as caption text
    - Creates 3 binary labels:
        sarcastic_bin, offensive_bin, motivational_bin

    loader.MemotionDataset(MEMOTION_DATA_PATH).data
    will give you a DataFrame with these extra columns.
    """

    def __init__(self, data_path):
        """
        Args:
            data_path: either
              - a directory containing 'labels.csv', or
              - the full path to the CSV file itself.
        """
        super().__init__()

        data_path = Path(data_path)

        if data_path.is_dir():
            csv_path = data_path / "labels.csv"
        else:
            csv_path = data_path

        self.data = self._load_data(csv_path)

    def _load_data(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            csv_path,
            dtype={"id": str, "text_corrected": str},
            low_memory=False,
            quotechar='"',
        )

        # Drop missing IDs or captions
        df = df[pd.notna(df["id"])]
        df = df[pd.notna(df["text_corrected"])]

        # --- Map sarcasm/offensive/motivational to binary labels ---

        # sarcastic_bin: 1 if any sarcasm other than "not_sarcastic"
        df["sarcastic_bin"] = df["sarcasm"].apply(
            lambda x: 0 if str(x).strip().lower() == "not_sarcastic" else 1
        )

        # offensive_bin: 1 if any offensive label other than "not_offensive"
        df["offensive_bin"] = df["offensive"].apply(
            lambda x: 0 if str(x).strip().lower() == "not_offensive" else 1
        )

        # motivational_bin: 1 if exactly "motivational"
        df["motivational_bin"] = df["motivational"].apply(
            lambda x: 1 if str(x).strip().lower() == "motivational" else 0
        )

        # Reset index for clean __getitem__
        df = df.reset_index(drop=True)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = row["text_corrected"]

        labels = np.array(
            [
                row["sarcastic_bin"],
                row["offensive_bin"],
                row["motivational_bin"],
            ],
            dtype=np.float32,
        )

        sample = {
            "id": row["id"],
            "image_name": row.get("image_name", None),
            "text": text,
            "labels": torch.from_numpy(labels),
            "humour": row.get("humour", None),
            "sarcasm": row.get("sarcasm", None),
            "offensive": row.get("offensive", None),
            "motivational": row.get("motivational", None),
            "overall_sentiment": row.get("overall_sentiment", None),
        }

        return sample
