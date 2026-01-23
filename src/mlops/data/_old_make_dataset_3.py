import os
import shutil
import logging
from pathlib import Path
from xmlrpc import client
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import hydra
import omegaconf
from omegaconf import DictConfig
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

log = logging.getLogger(__name__)

def download_data(cfg: DictConfig) -> Path:
    """
    Download raw dataset from a public GCS bucket to a local directory.

    Reads from:
      gs://{raw_bucket}/{raw_prefix}/

    Writes to:
      {cfg.data.raw_dir}/
    """

    # Resolve local destination (Hydra-safe)
    raw_data_path = Path(hydra.utils.to_absolute_path(cfg.cloud.data_gcs.bucket))

    raw_bucket = cfg.cloud.data_gcs.bucket
    raw_prefix = cfg.cloud.data_gcs.raw_dir

    log.info("DOWNLOADING RAW DATA FROM GCS")
    log.info(f"  bucket : {raw_bucket}")
    log.info(f"  prefix : {raw_prefix}")
    log.info(f"  target : {raw_data_path}")

    # Skip download if data already exists
    if raw_data_path.exists() and (raw_data_path / "Train.csv").exists():
        log.info("Raw data already present locally. Skipping download.")
        return raw_data_path

    # Anonymous client for public bucket (NO OAuth)
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(raw_bucket)

    # List blobs explicitly so we can validate
    blobs = list(bucket.list_blobs(prefix=raw_prefix))
    log.info(f"FOUND {len(blobs)} OBJECTS IN BUCKET")

    if not blobs:
        raise RuntimeError(
            f"No objects found in gs://{raw_bucket}/{raw_prefix}. "
            "Check bucket name or prefix."
        )
    log.info("STARTING DOWNLOAD OF RAW DATA...")
    # Ensure local directory exists
    raw_data_path.mkdir(parents=True, exist_ok=True)

    log.info("Checkpoint 1")
    log.info(blobs[0].name)
    # Download files
    for blob in blobs:
        log.info(f"Processing blob: {blob.name}")
        # Remove prefix from object path
        relative_path = Path(blob.name).relative_to(raw_prefix)
        local_file_path = raw_data_path / relative_path

        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_file_path)

    log.info(f"RAW DATA DOWNLOADED SUCCESSFULLY TO {raw_data_path}")
    return raw_data_path


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
    log.info("Data split complete.")
    # 3. Save
    output_dir.mkdir(parents=True, exist_ok=True)

    train_save_path = output_dir / "train_split.csv"
    val_save_path = output_dir / "val_split.csv"

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    log.info(f"Saved train split to {train_save_path}")
    # Copy test.csv as is for convenience
    if test_csv_path.exists():
        shutil.copy(test_csv_path, output_dir / "test.csv")

    log.info(f"Saved splits to {output_dir}")
    log.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

def upload_processed_data(local_dir: Path, cfg: DictConfig):
    processed_bucket = cfg.cloud.data_gcs.bucket
    processed_prefix = cfg.cloud.data_gcs.processed_dir

    log.info("UPLOADING PROCESSED DATA TO GCS")
    log.info(f"  bucket : {processed_bucket}")
    log.info(f"  prefix : {processed_prefix}")
    log.info(f"  source : {local_dir}")

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(processed_bucket)

    files = list(local_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No processed CSV files found in {local_dir}")

    for file in files:
        blob_path = f"{processed_prefix}/{file.name}"
        bucket.blob(blob_path).upload_from_filename(file)
        log.info(f"Uploaded {file.name} → gs://{processed_bucket}/{blob_path}")

    log.info("PROCESSED DATA UPLOAD COMPLETE")


@hydra.main(config_path="../../../configshydra", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Ensure processed path is absolute (Hydra changes working dir)
    processed_dir = Path(hydra.utils.to_absolute_path(cfg.cloud.data_gcs.processed_dir))

    # print(omegaconf.OmegaConf.to_yaml(cfg))
    # 1. Download & Clean Path
    clean_data_path = download_data(cfg)
    split_data(clean_data_path, processed_dir, cfg)
    upload_processed_data(processed_dir, cfg)



if __name__ == "__main__":
    """Ensure you have a file on Project Root level '.env'
        Following this exact format:
            KAGGLE_USERNAME="<KAGGLE USERNAME>"
            KAGGLE_KEY="<API KEY>"
        """

    main()
