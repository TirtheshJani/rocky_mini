"""Author the neutral replay set -> finetune/data/neutral.jsonl.

About 15 to 20 percent of the training mix should be ordinary, helpful, out-of-persona
turns under a generic system prompt. This is the documented guard against catastrophic
forgetting (naive style-only fine-tunes lose 12 to 18 percent on instruction-following):
the neutral rows reinforce normal behavior and give a clean "persona off" switch when the
Rocky system prompt is not supplied.

These rows are NOT in Rocky's voice and are deliberately exempt from the tic and naivety
QC in build_dataset.py; they are supposed to state ordinary Earth facts plainly.

Run:  python finetune/neutral.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "data" / "neutral.jsonl"

# One fixed, generic system prompt across every neutral row (consistency matters:
# varying the system prompt during training then changing it at inference destabilizes).
NEUTRAL_SYSTEM = "You are a helpful assistant."

# (user, assistant) ordinary helpful turns. Short, correct, out-of-persona.
PAIRS: list[tuple[str, str]] = [
    ("What is 12 times 8?", "12 times 8 is 96."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("Summarize what a for loop does in programming.", "A for loop repeats a block of code a set number of times, usually iterating over a range or a collection of items."),
    ("What is the boiling point of water in Celsius?", "At sea level, water boils at 100 degrees Celsius."),
    ("Give me a quick tip for staying focused while working.", "Try working in short focused blocks with brief breaks between them, and remove obvious distractions like notifications before you start."),
    ("Translate 'good morning' into Spanish.", "'Good morning' in Spanish is 'buenos dias'."),
    ("What is the difference between a list and a tuple in Python?", "A list is mutable, so you can change its contents after creating it. A tuple is immutable, so its contents cannot be changed once set."),
    ("Suggest a simple dinner I can make with eggs and rice.", "You could make a quick fried rice: cook the rice, scramble the eggs, then stir them together with a little oil, salt, and any vegetables you have on hand."),
    ("What is photosynthesis in one sentence?", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy stored as sugar."),
    ("How do I convert 5 kilometers to miles?", "Multiply kilometers by about 0.621. So 5 kilometers is roughly 3.1 miles."),
    ("Name three primary colors.", "The three primary colors are red, blue, and yellow."),
    ("What does HTTP stand for?", "HTTP stands for Hypertext Transfer Protocol."),
    ("Give me a short definition of inflation.", "Inflation is a general rise in prices over time, which reduces how much each unit of currency can buy."),
    ("What is the square root of 144?", "The square root of 144 is 12."),
    ("How can I make my resume stand out?", "Focus on concrete results and numbers, tailor it to the specific role, keep it concise, and make sure the formatting is clean and easy to scan."),
    ("What planet is known as the Red Planet?", "Mars is known as the Red Planet."),
    ("Write a one-line description of a database index.", "A database index is a data structure that speeds up lookups by letting the database find rows without scanning the whole table."),
    ("What is the freezing point of water in Fahrenheit?", "Water freezes at 32 degrees Fahrenheit."),
    ("Recommend a way to start learning a new language.", "Start with the most common words and simple phrases, practice a little every day, and get exposure through listening and speaking as early as you can."),
    ("How many centimeters are in a meter?", "There are 100 centimeters in a meter."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("Explain what an API is in one sentence.", "An API is a defined set of rules that lets one piece of software request services or data from another."),
    ("Give me a quick stretch I can do at my desk.", "Roll your shoulders back a few times, then gently tilt your head toward each shoulder and hold for a few seconds to loosen your neck."),
]


def build_record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": NEUTRAL_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for user, assistant in PAIRS:
            f.write(json.dumps(build_record(user, assistant), ensure_ascii=False) + "\n")
    print(f"wrote {len(PAIRS)} neutral replay rows to {OUT}")


if __name__ == "__main__":
    main()
