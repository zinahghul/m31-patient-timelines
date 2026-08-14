from pathlib import Path

import torch
from huggingface_hub import login
from src.model import EHRTransformer


# Configuration
MODEL_REPO = "zinahghulam/m31-patient-timelines"
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")


def main():
    # 1. Authenticate with Hugging Face.
    # This prompts for an access token and does not store the token in this file.
    login()

    # 2. Verify that the trained checkpoint exists.
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}\n"
            "Train the model first and ensure best_model.pt is saved "
            "inside the checkpoints/ directory."
        )

    # 3. Initialize the model architecture.
    model = EHRTransformer(num_classes=40)

    # 4. Load the trained weights.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint)

    # 5. Upload the model architecture and weights to Hugging Face.
    model.push_to_hub(MODEL_REPO)

    print(f"Model successfully uploaded to: {MODEL_REPO}")


if __name__ == "__main__":
    main()