import torch
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# BERT model
BERT_MODEL_NAME = "bert-base-uncased"
BERT_DIM = 768

# Processing parameters (you can later make these command-line arguments)
BATCH_SIZE = 128
DEFAULT_M = 120   # Monte Carlo samples per class (from paper)
DEFAULT_K = 50    # Nearest neighbors (from paper)

# Base data directory - change this or pass via argument
BASE_DATA_DIR = "data"   # Recommended: relative to repo root