import os
import json
import gdown
from concurrent.futures import ThreadPoolExecutor

os.makedirs("sentence_videos", exist_ok=True)
cat = json.load(open("catalog.json", encoding="utf-8"))

NEW_50 = [
    "All the best Good luck",
    "Hurry Up Quickly",
    "Railway Station",
    "Bus Stop",
    "AC Express Train",
    "Bullet Train",
    "Local Train",
    "Domestic Flight",
    "International Flight",
    "Boarding Pass",
    "First Aid",
    "Ambulance Care Assistant",
    "Blood Bank",
    "Blood Group",
    "Blood pressure",
    "Medical Certificate",
    "Birth Certificate",
    "Death Certificate",
    "Income Certificate",
    "Fitness Certificate",
    "Fire Station",
    "Customer Care",
    "Shopping Cart",
    "Voter ID card",
    "Key Card",
    "Memory Card",
    "Debit Card",
    "Bottled water",
    "Drinking Water",
    "Water Purifier",
    "Water tap",
    "Junk Food",
    "Food Poisoning",
    "Room Service",
    "Fitting Room",
    "Cold Room",
    "Front office",
    "Back office",
    "Part-Time Job",
    "Full Time",
    "Public Holiday",
    "Daily Everyday",
    "Next Day",
    "Close friend",
    "Clean Tidy",
    "Earbuds Earphones Headphones",
    "Charger (Phone)",
    "Electric Car",
    "Electric Bus",
    "Car Wash"
]

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
    out_path = os.path.join("sentence_videos", filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"Already exists: {filename}", flush=True)
        return

    url = f"https://drive.google.com/uc?id={fid}"
    print(f"Downloading {target} -> {filename}...", flush=True)
    gdown.download(url, out_path, quiet=True)
    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"Downloaded {filename} ({size} bytes)", flush=True)

def main():
    print(f"Starting concurrent download of {len(NEW_50)} new sentence videos...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(download_one, NEW_50)

    total_files = len([f for f in os.listdir("sentence_videos") if f.endswith(".mp4")])
    print(f"\nAll downloads completed! Total sentence videos in folder: {total_files}", flush=True)

if __name__ == "__main__":
    main()
