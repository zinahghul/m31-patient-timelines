from huggingface_hub import login
from src.model import EHRTransformer
import torch

# 1. Log in to Hugging Face (this will prompt you for your Access Token)
# You can generate a token at: https://huggingface.co/settings/tokens
login()

# 2. Initialize the model architecture with the correct number of classes
model = EHRTransformer(num_classes=40)

# 3. Load your trained weights
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# 4. Push directly to your profile
model.push_to_hub("zinahghulam/m31-patient-timelines")