import gdown
import json
import sys

# Force utf-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

url = "https://drive.google.com/drive/folders/1U-Pr4r1-cupgNOOq9NH_uTsQnPSVEKco"

print("Fetching file list with UTF-8 encoding...")
try:
    files = gdown.download_folder(url=url, skip_download=True, quiet=True)
    catalog = []
    for f in files:
        catalog.append({
            "id": f.id,
            "path": f.path,
        })
    with open("catalog.json", "w", encoding="utf-8") as out:
        json.dump(catalog, out, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(catalog)} files to catalog.json")
except Exception as e:
    print("Error:", e)
