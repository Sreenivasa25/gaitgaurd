#!/usr/bin/env python3

import os
import sys

print("🔍 GaitGuard Setup Verification\n")

checks = []

# 1. Python version
python_version = sys.version_info
print(
    "✓ Python version:",
    sys.version.split()[0],
)
checks.append(python_version >= (3, 10))

# 2. Virtual environment
venv_active = (
    hasattr(sys, "real_prefix")
    or (
        hasattr(sys, "base_prefix")
        and sys.base_prefix != sys.prefix
    )
)

print(
    "✓ Virtual environment:",
    "activated" if venv_active else "NOT active",
)
checks.append(venv_active)

# 3. Core package imports
try:
    import cv2
    import mediapipe
    import numpy
    import sklearn

    print("✓ Core packages: imported successfully")
    checks.append(True)

except ImportError as e:
    print(f"✗ Core packages: {e}")
    checks.append(False)

# 4. Folder structure
expected_dirs = [
    "src",
    "data",
    "tests",
    "experiments",
    "notebooks",
    "docs",
    "logs",
]

missing = [
    directory
    for directory in expected_dirs
    if not os.path.isdir(directory)
]

if not missing:
    print("✓ Folder structure: complete")
    checks.append(True)
else:
    print(f"✗ Missing: {', '.join(missing)}")
    checks.append(False)

# 5. Configuration
if os.path.exists(".env"):
    print("✓ Configuration: .env found")
    checks.append(True)
else:
    print(
        "⚠ Configuration: .env not found "
        "(copy from .env.example)"
    )
    checks.append(False)

print("\n" + "=" * 40)

if all(checks):
    print("✅ Setup complete! Ready to code.")
else:
    print("❌ Some checks failed. See above.")
    sys.exit(1)
