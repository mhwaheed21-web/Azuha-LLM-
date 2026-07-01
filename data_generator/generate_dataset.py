import json
import random

# Load profile
with open("profile.json", "r", encoding="utf-8") as f:
    profile = json.load(f)

templates = [
    {
        "instruction": "What's my name?",
        "output": f"Your name is {profile['name']}."
    },
    {
        "instruction": "Who am I?",
        "output": f"You are {profile['name']}, an {profile['degree']} student at {profile['university']}."
    },
    {
        "instruction": "What am I studying?",
        "output": f"You are studying {profile['degree']} at {profile['university']}."
    },
    {
        "instruction": "What are my interests?",
        "output": f"Your interests include {', '.join(profile['interests'])}."
    },
    {
        "instruction": "What skills do I have?",
        "output": f"You have experience with {', '.join(profile['skills'])}."
    },
    {
        "instruction": "What are my career goals?",
        "output": f"Your goals include becoming {', '.join(profile['career_goals'])}."
    }
]

dataset = []

for _ in range(10000):
    temp = random.choice(templates)

    dataset.append({
        "instruction": temp["instruction"],
        "input": "",
        "output": temp["output"]
    })

with open("hamza_assistant_10000.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("Dataset created successfully: hamza_assistant_10000.json")