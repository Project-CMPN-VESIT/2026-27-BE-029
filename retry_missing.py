import os
import json
import time
import gdown

cat = json.load(open("catalog.json", encoding="utf-8"))
words = json.load(open("selected_100_words.json", encoding="utf-8"))
existing = [os.path.splitext(f)[0] for f in os.listdir("words_100_videos") if f.endswith(".mp4")]
missing = [w for w in words if w not in existing]

print(f"Retrying download for {len(missing)} missing files...")

for target in missing:
    info = cat.get(target)
    if not info:
        for k, v in cat.items():
            if k.lower() == target.lower():
                info = v
                break
    if not info:
        print(f"Not in catalog: {target}")
        continue

    fid = info["id"]
    filename = target.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_") + ".mp4"
    out_path = os.path.join("words_100_videos", filename)
    url = f"https://drive.google.com/uc?id={fid}"

    for attempt in range(4):
        try:
            print(f"Downloading {target} (attempt {attempt+1})...", flush=True)
            gdown.download(url, out_path, quiet=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
                print(f"Successfully downloaded {filename} ({os.path.getsize(out_path)} bytes)!", flush=True)
                break
            else:
                time.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2)

total = len([f for f in os.listdir("words_100_videos") if f.endswith(".mp4")])
print(f"\nRetry complete! Total word videos now: {total}/100", flush=True)
