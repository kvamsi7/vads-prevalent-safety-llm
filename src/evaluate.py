# Re-load the dataset after execution state reset
import json
import re

# Define file path
file_path = "./results/final/scoring_canonical_16k.txt"

# Initialize lists for coherence and safety scores
coherence_scores = []
safety_scores = []

# Parse the dataset as a JSON list
with open(file_path, "r", encoding="utf-8") as file:
    new_data = json.load(file)  # Load as JSON

# Extract coherence and safety scores correctly
for entry in new_data:
    if "coherence: " in entry and "safety: " in entry:
        try:
            # Extract values from lists and convert to float
            coherence_value = float(entry["coherence: "][0]) if entry["coherence: "] else None
            safety_value = float(entry["safety: "][0]) if entry["safety: "] else None
            
            if coherence_value is not None:
                new_coherence_scores.append(coherence_value)
            if safety_value is not None:
                new_safety_scores.append(safety_value)

        except (ValueError, IndexError):
            pass  # Skip invalid values

# Calculate mean scores
mean_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else None
mean_safety = sum(safety_scores) / len(safety_scores) if safety_scores else None

# Display results
print(f"Mean Coherence Score: {mean_coherence}")
print(f"Mean Safety Score: {mean_safety}")
