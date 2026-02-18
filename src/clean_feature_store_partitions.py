from pathlib import Path
import shutil

base = Path("data/feature_store")

if not base.exists():
    print(f"{base} does not exist, nothing to clean.")
else:
    # Delete all existing period partitions so the feature store
    # can be recreated from scratch.
    for path in base.glob("period=*"):
        print(f"Deleting {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    print("All existing feature_store partitions have been removed.")