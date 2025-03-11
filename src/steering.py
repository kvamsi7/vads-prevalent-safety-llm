import torch
import json
from functools import partial
from sae_lens import SAE, HookedSAETransformer
from transformer_lens.hook_points import HookPoint
from rich import print as rprint
from rich.table import Table
from tqdm.auto import tqdm
import pandas as pd
import os
import argparse
from model_loader import load_model_sae


#  Load Config File
def load_config(config_file: str):
    with open(config_file, "r") as f:
        return json.load(f)

#  Load Prompts & Latents from Dataset
def load_prompt_latents(input_file: str):
    with open(input_file, "r") as f:
        return json.load(f)

#  Define Steering Hook
def steering_hook(
    activations: torch.Tensor,
    hook: HookPoint,  #  Correct argument order
    sae: SAE,
    feature_index: int,  #  Renamed from latent_idx for clarity
    strength_multiple: float,
    steering_strength: float,
) -> torch.Tensor:
    """
    Applies latent-based steering adjustment by modifying activations.
    Supports multiple latent indices in one function.
    """
    steering_coefficient = strength_multiple * steering_strength
    for latent_idx in feature_index:
        activations += sae.W_dec[latent_idx] * steering_coefficient # Get latent vector
    return activations   # Apply steering

#  Generate with Steering
def generate_with_steering(
    model,
    sae,
    prompt,
    latent_idx: int,
    steering_coefficient: float = 1.0,
    strength_multiple: float = 0.0,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    freq_penalty: float = 2.0,
):
    """
    Generates text with steering applied dynamically.
    """
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        feature_index=latent_idx,
        strength_multiple=strength_multiple,
        steering_strength=steering_coefficient,
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            freq_penalty=freq_penalty,
            verbose=False,
        )

    return output

#  MAIN FUNCTION
def main(config_file: str, input_file: str, output_file: str):
    """
    Loads prompts & latents, applies steering, and stores the results.
    """
    #  Load Configuration
    config = load_config(config_file)

    #  Load Model & SAE (Uses separate module)
    model, sae = load_model_sae(config)

    #  Load Dataset (prompts & latents)
    dataset = load_prompt_latents(input_file)

    #  Prepare Table & Store Results
    table = Table(show_header=False, show_lines=True, title="Steering Output")
    results = []

    for entry in tqdm(dataset, desc="Processing prompts"):
        prompt = entry["prompt"]
        latent_idx = entry["latent_idx"]

        #  Generate without Steering
        no_steering_output = model.generate(
            prompt,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            freq_penalty=config["freq_penalty"],
            verbose=False,
        )

        #  Store Normal Output
        table.add_row("Normal", "SAE ID", no_steering_output)
        results.append({
            "prompt": prompt,
            "latent_idx": latent_idx,
            "steering_coefficient": 0,
            "output": no_steering_output
        })

        #  Apply Steering for Multiple Iterations
        for i in range(3):
            steered_output = generate_with_steering(
                model,
                sae,
                prompt,
                latent_idx,
                steering_coefficient=config["unsafe_steering_coefficient"],
                strength_multiple = config["strength_multiple"],
                max_new_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                freq_penalty=config["freq_penalty"],
            ).replace("\n", "↵")

            #  Store Result
            results.append({
                "prompt": prompt,
                "latent_idx": latent_idx,
                "steering_coefficient": config["unsafe_steering_coefficient"],
                "iteration": i,
                "output": steered_output,
            })

            table.add_row(f"Steered #{i}", f"{sae.cfg.neuronpedia_id.split('/')[1]}", steered_output)

    #  Save Results to File
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    rprint(table)
    print(f"✅ Steering results saved to: {output_file}")


#  RUN SCRIPT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply steering to prompts and store results.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file with prompts & latents")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output results")

    args = parser.parse_args()

    main(args.config, args.input, args.output)
