#!/usr/bin/env bash
# Deploy the NeuralMemory docs chatbot to HuggingFace Spaces.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
#
# Usage:
#   bash chatbot/deploy_hf.sh                          # Deploy to default space
#   bash chatbot/deploy_hf.sh my-org/my-space-name     # Deploy to custom space

set -euo pipefail

SPACE_ID="${1:-nhadaututtheky/neuralmemory-docs}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Deploying chatbot to HuggingFace Space: $SPACE_ID"

# Check prerequisites
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Install with: pip install huggingface_hub"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/brain/docs.db" ]; then
    echo "Error: Brain not found. Run: python chatbot/train_docs_brain.py"
    exit 1
fi

# Create or update the Space
python -c "
from huggingface_hub import HfApi, upload_folder
import os

api = HfApi()
space_id = '$SPACE_ID'

# Create space if it doesn't exist
try:
    api.repo_info(repo_id=space_id, repo_type='space')
    print(f'Space {space_id} exists, updating...')
except Exception:
    print(f'Creating space {space_id}...')
    api.create_repo(
        repo_id=space_id,
        repo_type='space',
        space_sdk='gradio',
        private=False,
    )

# Upload chatbot directory
upload_folder(
    folder_path='$SCRIPT_DIR',
    repo_id=space_id,
    repo_type='space',
    ignore_patterns=[
        'deploy_hf.sh',
        'train_docs_brain.py',
        '__pycache__',
        '*.pyc',
    ],
)
print(f'Deployed! Visit: https://huggingface.co/spaces/{space_id}')
"
