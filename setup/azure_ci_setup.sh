#!/usr/bin/env bash
# One-time setup on an Azure ML compute instance (run from the repo root):
#   bash setup/azure_ci_setup.sh
# Creates a Python 3.12 conda env "fraud" with the pinned versions the model was trained with,
# registers it as a Jupyter kernel, validates the hand-off and runs the tests.
set -euo pipefail
cd "$(dirname "$0")/.."
source /anaconda/etc/profile.d/conda.sh
if ! conda env list | grep -qE '^fraud\s'; then
  # conda-forge only: avoids the Anaconda default-channel Terms of Service prompt
  conda create -n fraud -c conda-forge --override-channels python=3.12 pip -y -q
fi
conda activate fraud
pip install -q --upgrade pip
pip install -q -r requirements.txt ipykernel
python -m ipykernel install --user --name fraud --display-name "Python 3.12 (fraud)" >/dev/null
echo "== Workspace detected =="
python - <<'PY'
from mlops.azure_config import workspace_settings, endpoint_name
s = workspace_settings()
print(f"subscription={s['subscription_id']}\nresource_group={s['resource_group']}\nworkspace={s['workspace_name']}")
print(f"endpoint name will be: {endpoint_name()}")
PY
echo "== Hand-off check =="
python -m src.config
echo "== Tests =="
python -m pytest -q
echo
echo "Setup complete. Next:  conda activate fraud && python -m mlops.track_experiments --register"
