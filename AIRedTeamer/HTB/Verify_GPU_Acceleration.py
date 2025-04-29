import torch
import torch_directml

device = torch_directml.device()  # Get DirectML device
print(f"Using device: {device}")

# Test if DirectML works
try:
    x = torch.ones((2, 2), device=device)
    print("DirectML is working! 🎉")
except Exception as e:
    print(f"Error using DirectML: {e}")
