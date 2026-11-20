import json

with open("catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Look for exact single word or common 2-word phrases
word_list = [
    "Hello", "Namaste", "Good", "Bad", "Yes", "No", "Help", "Stop", "Wait", "Come",
    "Go", "Eat", "Drink", "Food", "Water", "Tea", "Milk", "Sleep", "Work", "Read",
    "Write", "Book", "Pen", "Friend", "Family", "Father", "Mother", "Brother", "Sister",
    "Son", "Daughter", "Baby", "Doctor", "Nurse", "Hospital", "School", "Teacher",
    "Student", "Morning", "Afternoon", "Evening", "Night", "Day", "Time", "Today",
    "Tomorrow", "Yesterday", "Happy", "Sad", "Angry", "Love", "Like", "Beautiful",
    "Clean", "Dirty", "Hot", "Cold", "Big", "Small", "Open", "Close", "Give", "Take",
    "Play", "Win", "Lose", "Money", "Phone", "Computer", "Car", "Bus", "Train",
    "Walk", "Run", "Stand", "Sit", "Laugh", "Cry", "Understand", "Sorry", "Please",
    "Welcome", "Thank You", "Name", "Where", "When", "Why", "What", "Who", "How"
]

exact_matches = {}
for word in word_list:
    # 1. exact match
    for k, v in catalog.items():
        if k.strip().lower() == word.lower():
            exact_matches[word] = v
            break
        # Or starts with word + space (e.g., 'Thank You')
        if k.strip().lower() == word.lower() + " (sign 1)" or k.strip().lower() == word.lower() + " (sign 2)":
            exact_matches[word] = v
            break

print(f"Found {len(exact_matches)} exact words:")
for i, (k, v) in enumerate(exact_matches.items()):
    print(f"[{i+1}] {k} -> file: {v['filename']} (id: {v['id']})")

with open("exact_50_isl_words.json", "w", encoding="utf-8") as out:
    json.dump(exact_matches, out, ensure_ascii=False, indent=2)

print(f"Saved {len(exact_matches)} exact words to exact_50_isl_words.json!")
