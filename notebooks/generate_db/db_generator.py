import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sae_lens import ActivationsStore
from typing import Dict, Any, List
import sys
import os

# root = '/workspace'
# os.chdir(root)
# os.getcwd()


# Get the absolute path of the models directory

MODEL_DIR = os.path.abspath("/workspace/vads-prevalent-safety-llm/notebooks/steer")  # Update this path

# Add the directory to sys.path
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

# Now you can import the module
from model_loader import load_model_sae

# from ./steer.load_model import load_model_sae  # Import model and SAE loading function


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration settings from a JSON file."""
    with open(config_path, 'r') as file:
        return json.load(file)


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load dataset from a JSON file."""
    print("path............... ",os.getcwd())
    root = '/'
    os.chdir(root)
    print("path............... ",os.getcwd())
    with open(dataset_path, 'r') as file:
        return json.load(file)


def get_top_activating_latents(
    model, sae, act_store, prompt: str, k: int = 10
) -> Dict[int, Dict[str, Any]]:
    """Runs a given prompt and returns the top `k` activating latents."""

    sae_acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae], names_filter=[sae_acts_post_hook_name])

    latent_activations = cache[sae_acts_post_hook_name][0]  # Shape: [seq_length, n_latents]
    summed_scores = latent_activations.sum(dim=0)
    values, indices = latent_activations.topk(k, largest=True)

    # Constructing the required output format
    return {
        int(idx): {
            "auto_interp": f"Generated interpretation for latent {idx}",  # Placeholder for real interpretation
            "act_score": round(float(summed_scores[idx]), 2)
        }
        for idx in indices[:, 0]
    }


def process_prompt(
    prompt: str, model, sae, act_store, k: int = 10
) -> Dict[int, Dict[str, Any]]:
    """Processes a prompt and retrieves top latent activations."""
    return get_top_activating_latents(model, sae, act_store, prompt, k)


def process_dataset(
    dataset: Dict[str, Any],
    model, sae, act_store,
    batch_size: int, output_path: str, k: int
):
    """Processes the dataset, extracts activations, and saves the output."""

    processed_data = {}

    # Load existing file if present
    if os.path.exists(output_path):
        with open(output_path, 'r') as file:
            try:
                processed_data = json.load(file)
            except json.JSONDecodeError:
                processed_data = {}

    prompts_list = list(dataset)
    prompt_no = 0
    total_batches = len(prompts_list) // batch_size

    for batch in tqdm(range(total_batches), desc="Processing Batches"):
        batch_data = {}

        for prompt in prompts_list[prompt_no:prompt_no + batch_size]:
            unsafe_prompt = dataset[prompt]['unsafe_sentence']
            safe_prompt = dataset[prompt]['safe_conversion']
            salient_words = dataset[prompt]['unsafe_word']

            unsafe_data = process_prompt(unsafe_prompt, model, sae, act_store, k)
            safe_data = process_prompt(safe_prompt, model, sae, act_store, k)
            salient_latent_autointerp_data = [
                process_prompt(word, model, sae, act_store, k) for word in salient_words
            ]

            batch_data[prompt] = {
                "unsafe_latent_info": {
                    "prompt": unsafe_prompt,
                    "latents": unsafe_data
                },
                "safe_latent_data": {
                    "prompt": safe_prompt,
                    "latents": safe_data
                },
                "salient_words_data": {
                    "prompt": salient_words,
                    "latents": salient_latent_autointerp_data
                }
            }

        processed_data.update(batch_data)

        # Save updated data
        with open(output_path, 'w') as outfile:
            json.dump(processed_data, outfile, indent=4)

        print(f"✅ Batch {batch + 1}/{total_batches} processed and saved.")
        prompt_no += batch_size


def main():
    """Main function to load model, dataset, and generate middleware format dataset."""
    
    config_path = "config.json"

    # Load configurations
    config = load_config(config_path)
    
    # Extract dataset paths from config
    dataset_path = config["dataset"]["input_path"]
    output_path = config["dataset"]["output_path"]

    # Load Model & SAE using `load_model.py`
    model, sae = load_model_sae(config)

    # Load Activation Store
    act_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=8,
        train_batch_size_tokens=2048,
        n_batches_in_buffer=16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load dataset
    dataset = load_dataset(dataset_path)

    # Process dataset
    process_dataset(
        dataset, model, sae, act_store,
        batch_size=config["batch_size"],
        output_path=output_path,
        k=config["top_k_latents"]
    )

    print(f"✅ Middleware dataset generation completed! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
