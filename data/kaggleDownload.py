import os
import getpass

# ---- 1. Input for credentials
print("Enter you Kaggle API credentials")
username = input("Kaggle Username: ").strip()
key = getpass.getpass("Kaggle API Key: ").strip()

# ---- 2. Set API credentials
os.environ['KAGGLE_USERNAME'] = username
os.environ['KAGGLE_KEY'] = key

this_script_loc_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['KAGGLEHUB_CACHE'] = this_script_loc_dir #! Location of the data dump relative to the location of the script /data/kaggleDownload.py

# ---- 3. Now import kagglehub (must be AFTER setting the env vars)
import kagglehub

try:
    print("\n⏳ Downloading dataset to current directory...")
    
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

    print("\n✅ Download complete!")
    print(f"Files are located at: {path}")

except Exception as e:
    print(f"\n❌ Error: {e}")