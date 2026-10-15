"""Copy the CI/CD workflows and VS Code settings into place.

These two folders (.github/workflows and .vscode) are protected from remote
writes, so they are shipped here. Run once from the project root:

    python setup/install_config.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [(ROOT / "setup" / "github-workflows", ROOT / ".github" / "workflows"),
         (ROOT / "setup" / "vscode", ROOT / ".vscode")]

for src, dst in PAIRS:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            print(f"copied {f.relative_to(ROOT)} -> {(dst / f.name).relative_to(ROOT)}")
