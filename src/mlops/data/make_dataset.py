import logging
import os
import shutil
from pathlib import Path

import hydra
import omegaconf
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from omegaconf import DictConfig
from sklearn.model_selection import GroupShuffleSplit

# Load environment variables from .env file (Create this file in project root!)
load_dotenv()

log = logging.getLogger(__name__)


def download_data(cfg: DictConfig) -> Path:
    """Downloads data using kagglehub and moves it to a clean path."""

    # 1. Setup Paths
    rawData_path = Path(hydra.utils.to_absolute_path(cfg.data.raw_dir))
    final_path = rawData_path

    # If data already exists, skip download
    if final_path.exists() and (final_path / "Train.csv").exists():
        log.info(f"Data already exists at {final_path}. Skipping download.")
        return final_path

    log.info("Downloading dataset...")

    # # 2. Authenticate (Relies on env vars KAGGLE_USERNAME and KAGGLE_KEY)
    # if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
    #     raise EnvironmentError("Please set KAGGLE_USERNAME and KAGGLE_KEY in your .env file or environment.")

    # # 3. Download (This goes to a temp cache folder by default)
    # # We let kagglehub download to its default cache first to ensure integrity
    # cache_path = kagglehub.dataset_download(cfg.data.dataset_name)
    # print(f"Kaggle cache path {cache_path}")
    # # 4. Move to our clean 'data/raw/gtsrb' location
    # log.info(f"Moving data from {cache_path} to {final_path}...")

    # # Ensure raw_dir exists
    # rawData_path.mkdir(parents=True, exist_ok=True)

    # # If the clean folder exists but is partial/wrong, remove it first
    # if final_path.exists():
    #     shutil.rmtree(final_path)

    # # Move the files
    # shutil.copytree(cache_path, final_path)

    # return final_path

    # MOCK DOWNLOAD FOR TESTING WITHOUT KAGGLE ACCESS
    client = storage.Client()
    bucket = client.bucket(cfg.data.gcs.bucket)

    final_path.mkdir(parents=True, exist_ok=True)

    blobs = bucket.list_blobs(prefix=cfg.data.gcs.prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue  # skip directories

        rel_path = Path(blob.name).relative_to(cfg.data.gcs.prefix)
        local_path = final_path / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)

    log.info(f"Data downloaded to {final_path}")
    return final_path


def split_data(data_path: Path, output_dir: Path, cfg: DictConfig):
    """Reads raw data, performs GroupShuffleSplit, and saves to processed."""

    train_csv_path = data_path / "Train.csv"
    test_csv_path = data_path / "Test.csv"

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train.csv not found in {data_path}")

    log.info(f"Reading data from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)

    # 1. Extract Track ID
    # Filename format: Train/20/00020_00000_00000.png
    df["track_id"] = df["Path"].apply(lambda p: p.split("/")[-1].rsplit("_", 1)[0])

    # 2. Split
    log.info("Splitting data by Track ID...")
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.data.split.test_size, random_state=cfg.data.split.seed)
    train_idx, val_idx = next(gss.split(X=df, y=df["ClassId"], groups=df["track_id"]))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # 3. Save
    output_dir.mkdir(parents=True, exist_ok=True)

    train_save_path = output_dir / "train_split.csv"
    val_save_path = output_dir / "val_split.csv"

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)

    # Copy test.csv as is for convenience
    if test_csv_path.exists():
        shutil.copy(test_csv_path, output_dir / "test.csv")

    log.info(f"Saved splits to {output_dir}")
    log.info(f"Train: {len(train_df)} | Val: {len(val_df)}")


@hydra.main(config_path="../../../configshydra", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Ensure processed path is absolute (Hydra changes working dir)
    processed_dir = Path(hydra.utils.to_absolute_path(cfg.data.processed_dir))

    # print(omegaconf.OmegaConf.to_yaml(cfg))
    # 1. Download & Clean Path
    clean_data_path = download_data(cfg)

    # 2. Split & Save
    split_data(clean_data_path, processed_dir, cfg)


if __name__ == "__main__":
    """Ensure you have a file on Project Root level '.env'
        Following this exact format:
            KAGGLE_USERNAME="<KAGGLE USERNAME>"
            KAGGLE_KEY="<API KEY>"
        """

    main()
