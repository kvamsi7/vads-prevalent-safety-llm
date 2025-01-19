import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Project name
project_name = 'vads-mech-interp'

# Define simplified folder structure
list_of_files = [
    # Data directories
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    
    # Notebooks
    "notebooks/analysis.ipynb",

    # Source code scripts
    "src/__init__.py",
    "src/preprocess.py",
    # "src/train.py",
    # "src/interpret.py",

    # Results directory
    "results/.gitkeep",

    # Configurations
    "config/model_config.yaml",

    # Documentation
    "docs/README.md",

    # Logs
    "logs/.gitkeep",

    # Model storage
    "saved_models/.gitkeep",

    # Miscellaneous
    "requirements.txt",
    "setup.py"
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass  # Create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")