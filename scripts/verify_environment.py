"""
Environment and Project Structure Verification Script
"""
import sys
from pathlib import Path

REQUIRED_DIRS = [
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/layouts",
    "frontend/src/services",
    "frontend/src/hooks",
    "frontend/src/utils",
    "frontend/src/data",
    "backend/app/api/routes",
    "backend/app/api/dependencies",
    "backend/app/core",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/utils",
    "backend/tests",
    "ml/datasets",
    "ml/preprocessing",
    "ml/training",
    "ml/models",
    "ml/inference",
    "ml/evaluation",
    "ml/explainability",
    "ml/notebooks",
    "data/raw",
    "data/processed",
    "data/external",
    "data/sample",
    "docs/architecture",
    "docs/dataset",
    "docs/models",
    "docs/research",
    "scripts",
    "tests"
]

def verify_structure(root: Path) -> bool:
    print(f"Verifying CycloneAI Project at: {root.resolve()}")
    all_ok = True
    for rdir in REQUIRED_DIRS:
        p = root / rdir
        if not p.is_dir():
            print(f"[FAIL] Missing directory: {rdir}")
            all_ok = False
        else:
            print(f"[OK] Directory exists: {rdir}")
            
    # Check key files
    key_files = [
        "README.md",
        "PROJECT_STATUS.md",
        ".gitignore",
        "backend/requirements.txt",
        "backend/app/main.py",
        "frontend/package.json"
    ]
    for kf in key_files:
        p = root / kf
        if not p.is_file():
            print(f"[FAIL] Missing file: {kf}")
            all_ok = False
        else:
            print(f"[OK] File exists: {kf}")
            
    return all_ok

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if verify_structure(project_root):
        print("\nAll required folders and key files are present.")
        sys.exit(0)
    else:
        print("\nSome directories or files are missing!")
        sys.exit(1)
