"""Utility script for uploading and downloading model checkpoints to/from Hugging Face Hub.

Added: a hard-coded download function as requested.
Notes:
 - Comments are in English.
 - Path, repo type, and repo id are hard-coded as per requirement.
 - Original simple upload call kept (commented) for reference; can be re-enabled if needed.
"""

from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


api = HfApi()

# Original upload snippet (disabled). Uncomment if you still need the upload behavior.
# api.upload_folder(
#     folder_path="../checkpoints",  # Local folder containing checkpoints
#     repo_id="SteveZh/momadiff_models",  # Repository ID (owner/repo_name)
#     repo_type="model",  # Repository type; use "model" for model repos
# )


def download_models():
    """Download a specific repo snapshot with hard-coded parameters.

    Hard-coded values (as required):
      repo_id: SteveZh/momadiff_models
      repo_type: model
      local_dir: ../checkpoints_downloaded

    This will download (or update) the repository snapshot into the target folder.
    Existing files will be reused (Hugging Face cache) unless changed.
    """
    # Hard-coded parameters
    repo_id = "SteveZh/momadiff_models"  # Fixed repository ID
    repo_type = "model"  # Fixed repository type
    local_dir = Path("../checkpoints_downloaded").resolve()  # Fixed local output path

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Downloading repo_id='{repo_id}' (type={repo_type}) to '{local_dir}' ...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # Make a full copy instead of symlinks for portability
        resume_download=True,
    )
    print(f"[Success] Download complete. Local path: {snapshot_path}")


if __name__ == "__main__":
    # Execute the hard-coded download when run as a script.
    download_models()


