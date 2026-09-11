import sys
from pathlib import Path

# Add finetune directory so `registration` is importable as a top-level package in tests
finetune_dir = Path(__file__).resolve().parent.parent.parent
if str(finetune_dir) not in sys.path:
    sys.path.insert(0, str(finetune_dir))
