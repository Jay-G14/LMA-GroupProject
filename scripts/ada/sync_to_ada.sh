#!/bin/bash
# Copy the code and SciFact data to Ada (~/LMA-Project), without git history, PUBHEALTH, local results or caches.
# Results come back with: rsync -az jayant.g@10.4.24.30:~/LMA-Project/to_transfer/ to_transfer/
set -e
cd "$(dirname "$0")/../.."
rsync -az --exclude '.git' --exclude 'data/PUBHEALTH' --exclude 'data/raw/scifact.tar.gz' --exclude 'results' \
      --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.venv' --exclude 'to_transfer' --exclude 'logs' \
      --exclude 'annotation' --exclude '.hf' ./ jayant.g@10.4.24.30:~/LMA-Project/
ssh jayant.g@10.4.24.30 'mkdir -p ~/LMA-Project/logs ~/LMA-Project/to_transfer'
echo "synced to jayant.g@10.4.24.30:~/LMA-Project"
