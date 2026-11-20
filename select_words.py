import json

with open("catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

print(f"Total catalog words: {len(catalog)}")

# Candidates for everyday signs
candidates = [
    "Hello", "Namaste", "Thank You", "Please", "Sorry", "Welcome", "Good", "Bad",
    "Yes", "No", "Help", "Stop", "Wait", "Come", "Go", "Eat", "Drink", "Water",
    "Food", "Tea", "Milk", "Sleep", "Wake Up", "Work", "Study", "Read", "Write",
    "Speak", "Hear", "See", "Understand", "Friend", "Family", "Mother", "Father",
    "Brother", "Sister", "Doctor", "Hospital", "School", "Home", "House", "Today",
    "Tomorrow", "Yesterday", "Time", "Morning", "Night", "Love", "Happy", "Sad",
    "Angry", "Beautiful", "Clean", "Open", "Close", "Give", "Take", "Play", "Money",
    "Phone", "Computer", "Book", "Car", "Bus", "Train", "Walk", "Run", "Stand", "Sit"
]

found = {}
lower_map = {k.lower(): k for k in catalog.keys()}

for cand in candidates:
    cl = cand.lower()
    if cl in lower_map:
        orig = lower_map[cl]
        found[orig] = catalog[orig]
    else:
        # Try finding prefix or match
        matches = [k for k in catalog.keys() if cl in k.lower()]
        if matches:
            orig = matches[0]
            found[orig] = catalog[orig]

print(f"Matched {len(found)} candidate words:")
for i, (k, v) in enumerate(list(found.items())[:55]):
    print(f"[{i+1}] {k}: id={v['id']}")

# If we have at least 50, select the top 50
selected = dict(list(found.items())[:50])
with open("selected_50.json", "w", encoding="utf-8") as out:
    json.dump(selected, out, ensure_ascii=False, indent=2)

print(f"\nSaved {len(selected)} selected words to selected_50.json!")
