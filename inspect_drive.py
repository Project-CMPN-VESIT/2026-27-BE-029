import gdown
import json

url = "https://drive.google.com/drive/folders/1U-Pr4r1-cupgNOOq9NH_uTsQnPSVEKco"

print("Listing files in Google Drive folder with skip_download=True...")
try:
    files = gdown.download_folder(url=url, skip_download=True, quiet=False)
    print(f"Total files found: {len(files)}")
    for i, f in enumerate(files[:30]):
        print(f"[{i+1}] path: {f.path}, id: {f.id}")
except Exception as e:
    print("Error:", e)
