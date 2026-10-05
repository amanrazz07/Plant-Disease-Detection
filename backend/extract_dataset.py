"""Extract the downloaded PlantVillage zip and organize into data/PlantVillage/."""
import zipfile
import shutil
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "PlantVillage"
TEMP_DIR = Path(__file__).parent / "data" / "_temp"
ZIP_PATH = TEMP_DIR / "plantvillage.zip"

print("Extracting PlantVillage dataset...")
print(f"  Zip: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.0f} MB)")

# Extract
extract_dir = TEMP_DIR / "extracted"
extract_dir.mkdir(parents=True, exist_ok=True)

print("  Extracting zip (this takes a minute)...")
with zipfile.ZipFile(str(ZIP_PATH), "r") as z:
    z.extractall(str(extract_dir))

print("  Extraction complete. Searching for color images directory...")

# The GitHub repo structure is:
# PlantVillage-Dataset-master/raw/color/<class_folders>
target = None
for root, dirs, files in os.walk(str(extract_dir)):
    basename = os.path.basename(root)
    if basename == "color":
        # Check it has class subdirectories
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if len(subdirs) > 10:
            target = root
            break

if target is None:
    # Try looking for directories that look like class names
    for root, dirs, files in os.walk(str(extract_dir)):
        plant_dirs = [d for d in dirs if "___" in d]
        if len(plant_dirs) > 10:
            target = root
            break

if target is None:
    print("  ERROR: Could not find class directories in extracted data!")
    print("  Contents of extract dir:")
    for item in extract_dir.iterdir():
        print(f"    {item.name}")
        if item.is_dir():
            for sub in list(item.iterdir())[:5]:
                print(f"      {sub.name}")
    exit(1)

print(f"  Found class images at: {target}")

# Count classes
class_dirs = [d for d in os.listdir(target) if os.path.isdir(os.path.join(target, d))]
print(f"  Classes found: {len(class_dirs)}")

# Copy to final location
if DATA_DIR.exists():
    shutil.rmtree(str(DATA_DIR))

print(f"  Copying to {DATA_DIR} ...")
shutil.copytree(target, str(DATA_DIR))

# Count total images
total = 0
for d in DATA_DIR.iterdir():
    if d.is_dir():
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.JPG")) + list(d.glob("*.png")) + list(d.glob("*.jpeg"))
        total += len(imgs)

print(f"\n  ✅ Dataset ready!")
print(f"     Location: {DATA_DIR}")
print(f"     Classes: {len(class_dirs)}")
print(f"     Total images: {total}")

# Cleanup zip and temp extraction
print("\n  Cleaning up temp files...")
shutil.rmtree(str(extract_dir), ignore_errors=True)
# Keep the zip in case needed, but delete extracted copy
print("  Done!")
