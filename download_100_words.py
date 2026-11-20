import os
import json
import gdown
from concurrent.futures import ThreadPoolExecutor

os.makedirs("words_100_videos", exist_ok=True)
cat = json.load(open("catalog.json", encoding="utf-8"))
words = json.load(open("selected_100_words.json", encoding="utf-8"))

def clean_filename(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_") + ".mp4"

def download_one(target):
    info = cat.get(target)
    if not info:
        for k, v in cat.items():
            if k.lower() == target.lower():
                info = v
                break
    if not info:
        print(f"Not found in catalog: {target}", flush=True)
        return

    fid = info["id"]
    filename = clean_filename(target)
    out_path = os.path.join("words_100_videos", filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"Already exists: {filename}", flush=True)
        return

    url = f"https://drive.google.com/uc?id={fid}"
    print(f"Downloading {target} -> {filename}...", flush=True)
    try:
        gdown.download(url, out_path, quiet=True)
        size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        print(f"Downloaded {filename} ({size} bytes)", flush=True)
    except Exception as e:
        print(f"Error downloading {target}: {e}", flush=True)

def main():
    print(f"Starting concurrent download of {len(words)} new word videos...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(download_one, words)

    total_files = len([f for f in os.listdir("words_100_videos") if f.endswith(".mp4")])
    print(f"\nAll downloads completed! Total word videos in folder: {total_files}", flush=True)

if __name__ == "__main__":
    main()
