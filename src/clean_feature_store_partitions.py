"""
Utility script to delete all existing period partitions from the feature store directory
so that it can be safely rebuilt from scratch.
"""

from pathlib import Path
import shutil

base = Path("data/feature_store")

if not base.exists():
    print(f"{base} does not exist, nothing to clean.")
else:
    # Iterate through 'period=*' partitions and remove them so a fresh
    # feature store can be generated without mixing old and new data.
    for path in base.glob("period=*"):
        print(f"Deleting {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    print("All existing feature_store partitions have been removed.")