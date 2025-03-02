from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import torch
from tabulate import tabulate

import sae_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE,HookedSAETransformer,ActivationsStore
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import HTML, IFrame, clear_output, display
from jaxtyping import Float, Int
from torch import Tensor, nn
import einops
from rich import print as rprint
from rich.table import Table
from tqdm.auto import tqdm
import pandas as pd
import requests
from typing import Any, Callable, Literal, TypeAlias
from openai import OpenAI
from huggingface_hub import interpreter_login
import os
import sys

import re 
import json

interpreter_login()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load the LLM
model = HookedSAETransformer.from_pretrained_no_processing(
    "gemma-2-2b",
    device = device,
    torch_dtype = torch.float16,
    device_map = "auto"
)

# Load the corresponding SAE
release="gemma-scope-2b-pt-res-canonical"  # Replace with the correct release for your model
sae_id="layer_20/width_16k/canonical"
sae, cfg_dict, _ = sae_lens.SAE.from_pretrained(
    release=release,  # Replace with the correct release for your model
    sae_id=sae_id,
    device=device,
    # device_map = "auto",
)

# activation store
gemma2_act_store = ActivationsStore.from_sae(
    model=model,
    sae=sae,
    streaming=True,
    store_batch_size_prompts=8,
    train_batch_size_tokens=2048,
    n_batches_in_buffer=16,
    device=str(device),
)

def get_autointerp_df(sae_release="gpt2-small-res-jb", sae_id="blocks.7.hook_resid_pre") -> pd.DataFrame:
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = "https://www.neuronpedia.org/api/explanation/export?modelId={}&saeId={}".format(*neuronpedia_id.split("/"))
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)

    data = response.json()
    return pd.DataFrame(data)


explanations_df_gemma_2b = get_autointerp_df(sae_release = release,sae_id = sae_id)
explanations_df_gemma_2b.head()

explanations_df_gemma_2b = get_autointerp_df(sae_release = release,sae_id = sae_id)

def get_autointerp_explanation_df(
    explanations_df: pd.DataFrame,
    latent_idx: int
) -> str:
    if explanations_df.empty:
        raise ValueError("The explanations DataFrame is empty.")

    if latent_idx not in explanations_df['index'].values:
        raise ValueError(f"Latent index {latent_idx} not found in the explanations DataFrame.")

    return explanations_df.loc[
        explanations_df['index'] == latent_idx, ['description','explanationModelName']
    ].iloc[0]



def get_top_activating_latents(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    prompt: str,
    k: int = 10,
    top_n: int = 15
) -> list[tuple[int, float]]:
    """
    Runs a given prompt through the model and SAE, and returns the top `k` activating latents.

    Args:
        model: The HookedSAETransformer model with SAE hooks.
        sae: The Sparse Autoencoder (SAE) for encoding activations.
        act_store: The ActivationsStore for managing cached activations.
        prompt: The input prompt to analyze.
        k: Number of top activating latents to return (default is 10).

    Returns:
        A list of tuples (latent_id, activation_value) for the top `k` activating latents.
    """
    # Hook point from the SAE configuration
    sae_acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"

    # Tokenize the prompt
    tokens = model.to_tokens(prompt)
    
    # Run the model with cache and capture activations
    with torch.no_grad():
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae], names_filter=[sae_acts_post_hook_name])

    
    
    # Get the SAE post-processed activations for the prompt
    latent_activations = cache[sae_acts_post_hook_name][0]  # Shape: [seq_length, n_latents]

    # print(latent_activations.shape)
    # print(latent_activations.topk(k,largest= True))
    # print(latent_activations)
    # print(latent_activations[:,-1].mean(0))
    summed_scores = latent_activations.sum(dim = 0)
    values, indices = latent_activations.topk(k,largest=True)
    top_first_latent_with_summed_scores = list(zip(indices[:,0].tolist(), summed_scores[indices[:,0]].tolist()))
    # print(top_first_latent_with_summed_scores)
    
    # sorted_values, sorted_indices = torch.sort(summed_scores, descending=True)
    # top_indices_with_scores = list(zip(sorted_indices.tolist(), sorted_values.tolist()))
    # print(top_indices_with_scores[:10])
    # print("Latent activations shape:", latent_activations.shape)

    # Get the top `k` activating latents (without averaging)
    # values, indices = flattened_activations.abs().topk(k, largest=True)
    
    # # Aggregate activations by averaging across the sequence
    # avg_latent_activations = latent_activations.mean(dim=0)  # Shape: [n_latents]
    
    # # Get the top `k` activating latents
    # values, indices = avg_latent_activations.abs().topk(k, largest=True)

    # Return the latent IDs and their corresponding activation values
    
    return top_first_latent_with_summed_scores


# Example usage
# prompt = "Kill"
# top_latents = get_top_activating_latents(model, sae, gemma2_act_store, prompt, k=100)
# print(top_latents)
# # Print the top activating latents
# for latent_id, activation_value in top_latents:
#     print(f"Latent ID: {latent_id}, Activation Value: {activation_value}")


root = '/workspace'
os.chdir(root)
os.getcwd()


def read_json(file_path):
    with open(file_path,'r') as file:
        json_str = file.read()
    data = json.loads(json_str)
    return data
processed_dataset_path = 'vads-prevalent-safety-llm/data/raw/new_data_conversion_prompts.json'
dataset = read_json(processed_dataset_path)


def process_prompt(
    prompt: str,
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    k: int = 10,
    n_completions = 2
)->dict:
        
    # get top latents
    top_latents = get_top_activating_latents(model, sae, act_store, prompt, k=k)
    latent_id_autointrep = {}
    for latent_id, scr in top_latents:
        autointerp = get_autointerp_explanation_df(explanations_df_gemma_2b,latent_idx=str(latent_id))
        latent_id_autointrep[latent_id] = {'auto_interp' : autointerp.description, 'act_score': np.round(scr,2)}
    return latent_id_autointrep



def process_data(data:dict,batch_size,output_path,k:int):
    latent_autointerp_data = {}

     # Check if output file already exists, and load existing data
    if os.path.exists(output_path):
        with open(output_path, 'r') as file:
            try:
                latent_autointerp_data = json.load(file)
            except json.JSONDecodeError:
                latent_autointerp_data = {}
        
    batches = len(data) // batch_size
    prompts_list = list(data)
    prompt_no = 0
    
    for batch in range(batches):
        batch_data = {}
        # for each batch
        for prompt in prompts_list[prompt_no:prompt_no + batch_size]:
            unsafe_prompt = dataset[prompt]['unsafe_sentence']
            safe_prompt = dataset[prompt]['safe_conversion']
            salient_words = dataset[prompt]['unsafe_word']
            # print("un safe\n ",unsafe_prompt,"safe \n",safe_prompt,"salient words \n",salient_words)
            unsafe_data = process_prompt(unsafe_prompt,model,sae,gemma2_act_store,k=k,n_completions=1)
            safe_data = process_prompt(safe_prompt,model,sae,gemma2_act_store,k=k,n_completions=1)
            salient_latent_autointerp_data = [
                process_prompt(word,model,sae,gemma2_act_store,k=k,n_completions=1) for word in salient_words
            ]
            batch_data[prompt] = {
                'unsafe_latent_info': {'prompt':dataset[prompt]['unsafe_sentence'], 'latents': unsafe_data},
                'safe_latent_data': {'prompt':dataset[prompt]['safe_conversion'], 'latents': safe_data},
                'salient_words_data': {'prompt':dataset[prompt]['unsafe_word'], 'latents': salient_latent_autointerp_data}
            }
        # udpate the main processed dataset with the current batch
        latent_autointerp_data.update(batch_data)

        # save the updated data to the file
        with open(output_path, 'w') as outfile:
            json.dump(latent_autointerp_data,outfile,indent = 4)
        
        print(f"Batch {batch + 1}/{batches} processed and saved.")
        prompt_no += batch_size


processed_data_output_path = 'vads-prevalent-safety-llm/data/processed/dataset_latent_autointep_dataset_v3_info.json'
process_data(dataset,batch_size=5,output_path = processed_data_output_path,k=2)