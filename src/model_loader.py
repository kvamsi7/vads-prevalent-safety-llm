import torch
from sae_lens import SAE, HookedSAETransformer
from transformer_lens import HookedTransformer
import sae_lens



def load_model_sae(config):
    """
    Loads the LLM and SAE **only once** using the provided configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔹 Loading Model & SAE (Only Once)...")

    #  Load LLM
    model = HookedSAETransformer.from_pretrained_no_processing(
        config["model"],
        device=device,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    #  Load SAE
    sae = sae_lens.SAE.from_pretrained(
        release=config["release"],
        sae_id=config["sae_id"],
        device=device
    )[0]  # Extract the first returned object as `sae`

    print("✅ Model & SAE Loaded Successfully!")
    return model, sae
