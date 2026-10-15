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
# PyPI can be slow from some Azure networks: long timeout + retries; re-running the script resumes
export PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10
echo "== PyPI reachability =="
curl -s -o /dev/null -w "pypi.org responded in %{time_total}s (HTTP %{http_code})\n" --max-time 60 https://pypi.org/simple/pip/ || echo "pypi.org not reachable"
pip install -q --upgrade pip
pip install -q -r requirements.txt ipykernel
python -m ipykernel install --user --name fraud --display-name "Python 3.12 (fraud)" >/dev/null
# Save exported workspace values to config.json once, so later terminals and scripts find them
if [ ! -f config.json ] && [ -n "${AZURE_SUBSCRIPTION_ID:-}" ] && [ -n "${AZURE_RESOURCE_GROUP:-}" ] && [ -n "${AZURE_ML_WORKSPACE:-}" ]; then
  printf '{\n  "subscription_id": "%s",\n  "resource_group": "%s",\n  "workspace_name": "%s"\n}\n' \
    "$AZURE_SUBSCRIPTION_ID" "$AZURE_RESOURCE_GROUP" "$AZURE_ML_WORKSPACE" > config.json
  echo "Saved workspace settings to config.json (git-ignored)"
fi
echo "== Workspace detected =="
[ -f /mnt/azmnt/.nbvm ] && grep -iE "^(subscription|resourcegroup|workspace)" /mnt/azmnt/.nbvm | sed 's/=.*/=<found>/' || echo "(no /mnt/azmnt/.nbvm)"
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
