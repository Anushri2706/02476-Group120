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
import gcsfs

load_dotenv()

log = logging.getLogger(__name__)

# def download_data(cfg: DictConfig) -> Path:
#     """
#     Download raw dataset from a public GCS bucket to a local directory.

#     Reads from:
#       gs://{raw_bucket}/{raw_prefix}/

#     Writes to:
#       {cfg.data.raw_dir}/
#     """

#     # Resolve local destination (Hydra-safe)
#     data_gcs_bucket_Path = Path(hydra.utils.to_absolute_path(cfg.cloud.data_gcs.bucket)) #Bucket path 

#     data_gcs_bucket = cfg.cloud.data_gcs.bucket
#     data_gcs_raw_dir = cfg.cloud.data_gcs.raw_dir
#     data_gcd_processed_dir = cfg.cloud.data_gcs.processed_dir

#     log.info("DOWNLOADING RAW DATA FROM GCS")
#     log.info(f"  bucket : {data_gcs_bucket}")
#     log.info(f"  prefix : {data_gcs_raw_dir}")
#     log.info(f"  target : {data_gcs_bucket_Path}")

#     # Skip download if data already exists
#     if data_gcs_bucket_Path.exists() and (data_gcs_bucket_Path / "Train.csv").exists():
#         log.info("Raw data already present locally. Skipping download.")
#         return data_gcs_bucket_Path

#     # Anonymous client for public bucket (NO OAuth)
#     client = storage.Client.create_anonymous_client()
#     bucket = client.bucket(data_gcs_bucket)

#     # List blobs explicitly so we can validate
#     blobs = list(bucket.list_blobs(prefix=data_gcs_raw_dir))
#     log.info(f"FOUND {len(blobs)} OBJECTS IN BUCKET")

#     if not blobs:
#         raise RuntimeError(
#             f"No objects found in gs://{data_gcs_bucket}/{data_gcs_raw_dir}. "
#             "Check bucket name or prefix."
#         )
#     log.info("STARTING DOWNLOAD OF RAW DATA...")
#     # Ensure local directory exists
#     data_gcs_bucket_Path.mkdir(parents=True, exist_ok=True)

#     log.info("Checkpoint 1")
#     log.info(blobs[0].name)
#     # Download files
#     for blob in blobs:
#         log.info(f"Processing blob: {blob.name}")
#         # Remove prefix from object path
#         relative_path = Path(blob.name).relative_to(data_gcs_raw_dir)
#         local_file_path = data_gcs_bucket_Path / relative_path

#         local_file_path.parent.mkdir(parents=True, exist_ok=True)
#         blob.download_to_filename(local_file_path)

#     log.info(f"RAW DATA DOWNLOADED SUCCESSFULLY TO {data_gcs_bucket_Path}")
#     return data_gcs_bucket_Path


def split_data(data_path: Path, output_dir: Path, bucket_name, cfg: DictConfig):
    """Reads raw data, performs GroupShuffleSplit, and saves to processed."""

    train_csv_path = f"gs://{bucket_name}/{cfg.cloud.data_gcs.raw_dir}/Train.csv"
    test_csv_path = f"gs://{bucket_name}/{cfg.cloud.data_gcs.raw_dir}/Test.csv"

    # if not train_csv_path.exists():
    #     raise FileNotFoundError(f"Train.csv not found in {data_path}")

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_save_path = os.path.join(output_dir, "train_split.csv")
    val_save_path = os.path.join(output_dir, "val_split.csv")

    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    log.info(f"Saved train split to {train_save_path}")
    # Copy test.csv as is for convenience
    # if test_csv_path.exists():
    #     shutil.copy(test_csv_path, os.path.join(output_dir, "test.csv"))

    log.info(f"Saved splits to {output_dir}")
    log.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

# def upload_processed_data(bucket_name, PATH_raw, DIR_processed, cfg: DictConfig):
#     log.info("UPLOADING PROCESSED DATA TO GCS")

#     client = storage.Client.create_anonymous_client()
#     bucket_obj = client.bucket(bucket_name)

#     files = list(PATH_raw.glob("*.csv"))
#     # if not files:
#     #     raise RuntimeError(f"No processed CSV files found in {PATH_raw}")
    
#     for file in files:
#         blob_path = f"{DIR_processed}/{file.name}"
#         bucket_obj.blob(blob_path).upload_from_filename(file)
#         log.info(f"Uploaded {file.name} → gs://{bucket_name}/{blob_path}")

def upload_processed_data(bucket_name, local_dir, gcs_dir, cfg=DictConfig):
    log.info("UPLOADING PROCESSED DATA TO GCS (via gcsfs)")

    local_dir = Path(local_dir)
    fs = gcsfs.GCSFileSystem()  

    files = list(local_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No processed CSV files found in {local_dir}")

    for file in files:
        gcs_path = f"{bucket_name}/{gcs_dir}/{file.name}"

        fs.put(str(file), gcs_path, overwrite=True)

        log.info(f"Uploaded {file.name} → gs://{gcs_path}")


# def upload_processed_data(local_dir: Path, cfg: DictConfig):
#     # processed_prefix = cfg.cloud.data_gcs.processed_dir

#     log.info("UPLOADING PROCESSED DATA TO GCS")
#     log.info(f"  bucket : {processed_bucket}")
#     log.info(f"  prefix : {processed_prefix}")
#     log.info(f"  source : {local_dir}")

#     client = storage.Client.create_anonymous_client()
#     bucket = client.bucket(processed_bucket)

#     files = list(local_dir.glob("*.csv"))
#     if not files:
#         raise RuntimeError(f"No processed CSV files found in {local_dir}")

#     for file in files:
#         blob_path = f"{processed_prefix}/{file.name}"
#         bucket.blob(blob_path).upload_from_filename(file)
#         log.info(f"Uploaded {file.name} → gs://{processed_bucket}/{blob_path}")

#     log.info("PROCESSED DATA UPLOAD COMPLETE")


@hydra.main(config_path="../../../configshydra", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Ensure processed path is absolute (Hydra changes working dir)
    bucket_name = cfg.cloud.data_gcs.bucket
    PATH_bucket = Path(hydra.utils.to_absolute_path(cfg.cloud.data_gcs.bucket)) #Bucket path 
    
    DIR_raw = cfg.cloud.data_gcs.raw_dir
    PATH_raw = Path(os.path.join(PATH_bucket, DIR_raw))

    DIR_processed = cfg.cloud.data_gcs.processed_dir
    PATH_processed = Path(os.path.join(PATH_bucket, DIR_processed))


    # print(omegaconf.OmegaConf.to_yaml(cfg))
    # 1. Download & Clean Path
    split_data(PATH_raw, PATH_processed, bucket_name, cfg)

    upload_processed_data(bucket_name=bucket_name,local_dir=PATH_processed, gcs_dir=DIR_processed,cfg=cfg)




if __name__ == "__main__":

    main()
