"""
run_quick_demo.py - Convenience script for a minimal demo run.
Equivalent to:
  python main.py --download-data --datasets ECG200 --trials 3 --trial-epochs 2 --final-epochs 5 --quick
"""

import subprocess
import sys

cmd = [
    sys.executable, "main.py",
    "--download-data",
    "--datasets", "ECG200",
    "--trials", "3",
    "--trial-epochs", "2",
    "--final-epochs", "5",
    "--quick",
]

print("Running quick demo...")
print("Command:", " ".join(cmd))
result = subprocess.run(cmd)
sys.exit(result.returncode)
