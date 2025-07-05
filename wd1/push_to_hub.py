# huggingface-cli upload diffusion-reasoning/sudoku-wll-neg-pos-1000 .
# huggingface-cli upload diffusion-reasoning/d1_us_sudoku-1000 .
# huggingface-cli upload diffusion-reasoning/countdown-wll-neg-pos-1000 .
# huggingface-cli upload diffusion-reasoning/d1_us_countdown-2500 .

import os

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))
print("OFFFF: ", os.getenv("HF_TOKEN"))
# Base path where your checkpoints are stored
base_model_path = "/home/rares/diffusion-rl/data/var_diff/checkpoints"

# Dataset checkpoints to upload
datasets = {
    "GSM8k": [1000, 2500, 5000, 7500],
    "MATH": [1000, 2500, 5000, 7500],
    "Sudoku": [1000, 2500, 4000, 5000],
    "Countdown": [1000, 2500, 4000],
}

names = {
    "GSM8k": "d1_SFT_us_gsm8k",
    "MATH": "d1_SFT_us_math",
    "Countdown": "d1_SFT_us_countdown",
    "Sudoku": "d1_SFT_us_sudoku",
}

# names = {
#     "GSM8k": "wll_SFT_NP_gsm8k",
#     "MATH": "wll_SFT_NP_math",
#     "Countdown": "wll_SFT_NP_countdown",
#     "Sudoku": "wll_SFT_NP_sudoku",
# }

# Loop through each dataset and its checkpoints
for dataset, checkpoints in datasets.items():
    for checkpoint in checkpoints:
        folder_path = os.path.join(
            base_model_path, names[dataset], f"checkpoint-{checkpoint}"
        )

        if os.path.exists(folder_path):
            repo_id = f"diffusion-reasoning/{names[dataset]}-{checkpoint}"

            # Create the repo if it doesn't exist
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    exist_ok=True,  # No error if it already exists
                    private=False,  # Change to True if you want private repos
                )
                print(f"Repo {repo_id} created or already exists.")
            except HfHubHTTPError as e:
                print(f"ERROR creating repo {repo_id}: {e}")
                continue  # Skip upload if repo creation fails

            # Upload the checkpoint folder
            print(f"Uploading {folder_path} to {repo_id}...")
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            print(f"WARNING: Path {folder_path} does not exist.")
