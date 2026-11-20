import re
import json

log_path = r"C:\Users\USER\.gemini\antigravity\brain\001be28b-ed85-482d-9a7d-0ee5852d8f88\.system_generated\tasks\task-157.log"

catalog = {}
pattern = re.compile(r"Processing file\s+([^\s]+)\s+(.+\.mp4)")

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            f_id = match.group(1).strip()
            f_name = match.group(2).strip()
            # Clean class name from file name (e.g., 'Hello.mp4' -> 'Hello')
            class_name = f_name.replace(".mp4", "").replace("_", " ").strip()
            catalog[class_name] = {
                "id": f_id,
                "filename": f_name,
                "class": class_name
            }

with open("catalog.json", "w", encoding="utf-8") as out:
    json.dump(catalog, out, ensure_ascii=False, indent=2)

print(f"Extracted {len(catalog)} unique ISL word videos to catalog.json!")
