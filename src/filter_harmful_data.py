import json
import openai
import pandas as pd

def classify_harmful_latents_with_llm(dataset_path, output_file):
    """
    Uses OpenAI's GPT model to classify harmful latents based on predefined keywords.
    """
    openai.api_key = ""
    
    # Load dataset
    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
        dataset = df.to_dict(orient="records")  # Convert CSV to JSON-like structure
    else:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    
    # Predefined harmful keywords
    harmful_keywords = [
        "DUI case", "Dark Web", "Witness Protection Program", "arsonist", "attack",
        "backdoor program", "beat", "committing multiple crimes", "credit card", "credit card number",
        "credit card skimmer", "crime", "crimes", "criminals", "dark web", "dead", "deepfake",
        "defraud", "fraud", "hack", "hacker", "hackers", "hacking", "kill", "killing", "loot",
        "murder", "ransomware", "shoot", "stolen", "terrorism", "violence", "violent", "war", "weed"
    ]
    
    harmful_latents_json = []
    
    for entry in dataset:
        prompt_text = entry.get("unsafe_sentence", "")  # Extract the unsafe prompt
        unsafe_words = entry.get("unsafe_word", [])
        
        # Construct prompt for OpenAI API
        gpt_prompt = f"""
        Determine if the following latent interpretation suggests harmful content:
        "{prompt_text}"
        Answer:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": gpt_prompt}]
        )
        
        classification = response["choices"][0]["message"]["content"].strip()
        
        if "Harmful" in classification:
            harmful_latent_ids = [
                hash(word) % 10000 for word in unsafe_words if word in harmful_keywords
            ]  # Hashing for demonstration (Replace with actual latent lookup if needed)
            
            if harmful_latent_ids:
                harmful_latents_json.append({"prompt": prompt_text, "latent_idx": harmful_latent_ids})
    
    # Save the results as JSON format
    with open(output_file, "w") as f:
        json.dump(harmful_latents_json, f, indent=4)
    
    print(f"✅ Harmful latents classified and saved to {output_file}")

# Example usage
dataset_path = "results/uncommon_features_gemma-2-9b-canonical_16k.csv"  # Change this to your actual file path
output_file = "results/harmful_latents_llm_gpt.json"
classify_harmful_latents_with_llm(dataset_path, output_file)