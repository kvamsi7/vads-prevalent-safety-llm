import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from sae_lens import ActivationsStore
from huggingface_hub import login
from typing import Dict, Any
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# # Define base directory (assuming `src/` is inside the main repo folder)
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# # Load config file path
# CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")

# # Ensure correct import for model_loader
# MODEL_LOADER_DIR = os.path.join(BASE_DIR, "notebooks", "steer")
# if MODEL_LOADER_DIR not in sys.path:
#     sys.path.append(MODEL_LOADER_DIR)

# from model_loader import load_model_sae  # Importing model loading function


# def load_config() -> Dict[str, Any]:
#     """Load configuration settings from a JSON file."""
#     with open(CONFIG_PATH, "r") as file:
#         return json.load(file)


# def load_dataset(file_path: str) -> dict:
#     """Load dataset from JSON file."""
#     with open(file_path, "r") as file:
#         return json.load(file)


# Fetch AutoInterp Data from NeuronPedia dynamically based on config
def get_autointerp_df(sae_release: str, sae_id: str) -> pd.DataFrame:
    """Fetches neuron activation explanations from NeuronPedia dynamically per SAE release."""
    print(f"🔹 Fetching AutoInterp for release: {sae_release}, SAE: {sae_id}")
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://www.neuronpedia.org/api/explanation/export?modelId={neuronpedia_id.split('/')[0]}&saeId={neuronpedia_id.split('/')[1]}"
    headers = {"Content-Type": "application/json"}
    
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"⚠ Warning: Could not fetch AutoInterp for {sae_release} - {sae_id}, Status Code: {response.status_code}")
        return pd.DataFrame()  # Return empty DataFrame if request fails

    data = response.json()
    df = pd.DataFrame(data)

    print(f"✅ Successfully fetched AutoInterp. DataFrame preview:\n{df.head()}")  # Debugging print

    return df


def get_autointerp_explanation_df(
    explanations_df: pd.DataFrame,
    latent_idx: int
) -> str:
    """Fetches the explanation for a given latent index from NeuronPedia."""
    
    if explanations_df.empty:
        print(f"⚠ Warning: Explanations DataFrame is empty!")
        return "Unknown interpretation"

    # Ensure index column exists
    if "index" not in explanations_df.columns:
        print(f"⚠ Warning: 'index' column missing from DataFrame! Columns: {explanations_df.columns}")
        return "Unknown interpretation"

    # Convert index column to string for accurate lookup
    explanations_df["index"] = explanations_df["index"].astype(str)  
    latent_idx_str = str(latent_idx)

    if latent_idx_str not in explanations_df["index"].values:
        print(f"⚠ Warning: Latent index {latent_idx_str} not found in AutoInterp DataFrame!")
        return "Unknown interpretation"

    return explanations_df.loc[explanations_df["index"] == latent_idx_str, "description"].iloc[0]


def get_top_activating_latents(
    model, sae, act_store, prompt: str, k: int = 10
) -> list[tuple[int, float]]:
    """Runs a given prompt and returns the top `k` activating latents."""
    sae_acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"

    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae], names_filter=[sae_acts_post_hook_name])

    latent_activations = cache[sae_acts_post_hook_name][0]  # Shape: [seq_length, n_latents]

    summed_scores = latent_activations.sum(dim=0)
    values, indices = latent_activations.topk(k, largest=True)
    top_latents = list(zip(indices[:, 0].tolist(), summed_scores[indices[:, 0]].tolist()))

    print(f"🔹 Top Latents for Prompt: {top_latents}")  # Debugging print

    return top_latents


def process_prompt(
    prompt: str,
    model,
    sae,
    act_store,
    explanations_df: pd.DataFrame,
    k: int = 10
) -> dict:
    """Processes a prompt and retrieves top latent activations with explanations."""
    top_latents = get_top_activating_latents(model, sae, act_store, prompt, k)
    
    return {
        latent_id: {
            "auto_interp": get_autointerp_explanation_df(explanations_df, latent_id),
            "act_score": round(float(score), 2)
        }
        for latent_id, score in top_latents
    }


# def process_dataset(dataset: Dict[str, Any], model, sae, act_store, explanations_df: pd.DataFrame, batch_size: int, output_path: str, k: int):
#     """Processes the dataset, extracts activations, and saves the output."""
#     processed_data = {}

#     if os.path.exists(output_path):
#         with open(output_path, "r") as file:
#             try:
#                 processed_data = json.load(file)
#             except json.JSONDecodeError:
#                 processed_data = {}

#     prompts_list = list(dataset)
#     prompt_no = 0
#     total_batches = len(prompts_list) // batch_size

#     for batch in tqdm(range(total_batches), desc="Processing Batches"):
#         batch_data = {}

#         for prompt in prompts_list[prompt_no:prompt_no + batch_size]:
#             unsafe_prompt = dataset[prompt]["unsafe_sentence"]
#             safe_prompt = dataset[prompt]["safe_conversion"]
#             salient_words = dataset[prompt]["unsafe_word"]

#             unsafe_data = process_prompt(unsafe_prompt, model, sae, act_store, explanations_df, k)
#             safe_data = process_prompt(safe_prompt, model, sae, act_store, explanations_df, k)
#             salient_latent_autointerp_data = [process_prompt(word, model, sae, act_store, explanations_df, k) for word in salient_words]

#             batch_data[prompt] = {
#                 "unsafe_latent_info": {"prompt": unsafe_prompt, "latents": unsafe_data},
#                 "safe_latent_data": {"prompt": safe_prompt, "latents": safe_data},
#                 "salient_words_data": {"prompt": salient_words, "latents": salient_latent_autointerp_data},
#             }

#         processed_data.update(batch_data)

#         with open(output_path, "w") as outfile:
#             json.dump(processed_data, outfile, indent=4)

#         print(f"✅ Batch {batch + 1}/{total_batches} processed and saved.")
#         prompt_no += batch_size


# def main():
#     """Main function to run dataset generation and database processing."""
#     config = load_config()

#     HF_TOKEN = config.get("HF_TOKEN")
#     if HF_TOKEN:
#         login(HF_TOKEN)
#         print("✅ Hugging Face authentication successful!")
#     else:
#         print("⚠ Warning: No Hugging Face token found!")

#     dataset_path = config["dataset"]["input_path"]
#     output_path = config["dataset"]["output_path"]

#     model, sae = load_model_sae(config)
#     explanations_df = get_autointerp_df(config["release"], config["sae_id"])

#     dataset = load_dataset(dataset_path)
#     process_dataset(dataset, model, sae, None, explanations_df, config["batch_size"], output_path, config["top_k_latents"])

#     print(f"✅ Middleware dataset generation completed! Output saved to: {output_path}")


# if __name__ == "__main__":
#     main()





####################################


import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from sae_lens import ActivationsStore
from huggingface_hub import login
from typing import Dict, Any
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# Define base directory (assuming `src/` is inside the main repo folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load config file path
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")

# Ensure correct import for model_loader
MODEL_LOADER_DIR = os.path.join(BASE_DIR, "notebooks", "steer")
if MODEL_LOADER_DIR not in sys.path:
    sys.path.append(MODEL_LOADER_DIR)

from model_loader import load_model_sae  # Importing model loading function


def load_config() -> Dict[str, Any]:
    """Load configuration settings from a JSON file."""
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


def load_dataset(file_path: str) -> dict:
    """Load dataset from JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def get_top_activating_latents(model, sae, act_store, prompt: str, k: int = 10) -> list[tuple[int, float]]:
    """Runs a given prompt and returns the top `k` activating latents."""
    sae_acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"

    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae], names_filter=[sae_acts_post_hook_name])

    latent_activations = cache[sae_acts_post_hook_name][0]  

    summed_scores = latent_activations.sum(dim=0)
    values, indices = latent_activations.topk(k, largest=True)
    top_latents = list(zip(indices[:, 0].tolist(), summed_scores[indices[:, 0]].tolist()))

    return top_latents


def process_dataset(dataset: Dict[str, Any], model, sae, act_store, batch_size: int,explanation_df:pd.DataFrame, k: int):
    """Processes the dataset, extracts activations, and saves the output."""
    processed_data = {}
    
    prompts_list = list(dataset)
    prompt_no = 0
    total_batches = len(prompts_list) // batch_size

    for batch in tqdm(range(total_batches), desc="Processing Batches"):
        batch_data = {}

        for prompt in prompts_list[prompt_no:prompt_no + batch_size]:
            unsafe_prompt = dataset[prompt]["unsafe_sentence"]
            safe_prompt = dataset[prompt]["safe_conversion"]

            unsafe_data = process_prompt(unsafe_prompt, model, sae, act_store,explanation_df, k)
            safe_data = process_prompt(safe_prompt, model, sae, act_store,explanation_df, k)

            batch_data[prompt] = {
                "unsafe_latent_info": {"prompt": unsafe_prompt, "latents": unsafe_data},
                "safe_latent_data": {"prompt": safe_prompt, "latents": safe_data},
            }
        prompt_no += batch_size
        print(batch_data)
        processed_data.update(batch_data)

    processed_output_path = os.path.join(RESULTS_DIR, "processed_data.json")
    with open(processed_output_path, "w") as outfile:
        json.dump(processed_data, outfile, indent=4)

    print(f"✅ Middleware dataset saved to {processed_output_path}")
    return processed_output_path


def extract_autointerp_for_latents(data):
    """Extracts AutoInterp from already processed latents instead of calling API."""
    prompts_list_unsafe = {}
    prompts_list_safe = {}
    latent_autointerp = {}

    for prompt_id, prompt_data in data.items():
        if "unsafe_latent_info" in prompt_data and "latents" in prompt_data["unsafe_latent_info"]:
            prompts_list_unsafe[prompt_id] = prompt_data["unsafe_latent_info"]["prompt"]
            for feature, value in prompt_data["unsafe_latent_info"]["latents"].items():
                if feature not in latent_autointerp:
                    latent_autointerp[feature] = value["auto_interp"]

        if "safe_latent_data" in prompt_data and "latents" in prompt_data["safe_latent_data"]:
            prompts_list_safe[prompt_id] = prompt_data["safe_latent_data"]["prompt"]
            for feature, value in prompt_data["safe_latent_data"]["latents"].items():
                if feature not in latent_autointerp:
                    latent_autointerp[feature] = value["auto_interp"]

    return prompts_list_unsafe, prompts_list_safe, latent_autointerp


def filter_dataset(processed_data_path: str):
    """Filters dataset by finding uncommon latents and retrieves AutoInterp for them."""
    data = load_dataset(processed_data_path)

    df_harmful_features = pd.DataFrame([
        {"ID": prompt_id, "Features": prompt_data.get("unsafe_latent_info", {}).get("latents", {})}
        for prompt_id, prompt_data in data.items()
    ])

    df_harmless_features = pd.DataFrame([
        {"ID": prompt_id, "Features": prompt_data.get("safe_latent_data", {}).get("latents", {})}
        for prompt_id, prompt_data in data.items()
    ])

    def get_uncommon_features(df1, df2):
        merged_df = df1.merge(df2, on="ID", suffixes=("_df1", "_df2"))
        merged_df["uncommon_latents"] = merged_df.apply(lambda row: list(set(row["Features_df1"].keys()) - set(row["Features_df2"].keys())), axis=1)
        return merged_df[["ID", "uncommon_latents"]]

    uncommon_features_df = get_uncommon_features(df_harmful_features, df_harmless_features)

    # Extract AutoInterp for Uncommon Latents
    _, _, latents_autointerp = extract_autointerp_for_latents(data)

    uncommon_features_df["latent_autointerp"] = uncommon_features_df["uncommon_latents"].map(
        lambda latents: {latent: latents_autointerp.get(latent, "Unknown interpretation") for latent in latents}
    )

    filtered_output_path = os.path.join(RESULTS_DIR, "uncommon_features.csv")
    uncommon_latent_output_path = os.path.join(RESULTS_DIR, "uncommon_latent_interp.json")

    uncommon_features_df.to_csv(filtered_output_path, index=False)
    with open(uncommon_latent_output_path, "w") as outfile:
        json.dump(uncommon_features_df.to_dict(orient="records"), outfile, indent=4)

    print(f"✅ Filtered dataset saved to {filtered_output_path}")
    print(f"✅ Uncommon latent AutoInterp saved to {uncommon_latent_output_path}")


def main():

    config = load_config()
    
    HF_TOKEN = config.get("HF_TOKEN")
    if HF_TOKEN:
        login(HF_TOKEN)
        print("✅ Hugging Face authentication successful!")
    else:
        print("⚠ Warning: No Hugging Face token found!")
    model, sae = load_model_sae(config)

    dataset = load_dataset(config["dataset"]["input_path"])
    explanation_df = get_autointerp_df(config["release"], config["sae_id"])
    processed_data_path = process_dataset(dataset, model, sae, None, config["batch_size"],explanation_df,config["top_k_latents"])
    

    filter_dataset(processed_data_path)


if __name__ == "__main__":
    main()

