import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA version (pytorch): {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
