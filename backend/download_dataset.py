"""
Download and prepare the PlantVillage dataset.

Tries multiple sources:
1. Kaggle API (requires ~/.kaggle/kaggle.json)
2. Direct download from GitHub mirror
3. Manual instructions if all else fails
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "PlantVillage"
TEMP_DIR = Path(__file__).parent / "data" / "_temp"


def try_kaggle():
    """Download via Kaggle CLI."""
    print("Trying Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(
            "emmarex/plantdisease",
            path=str(TEMP_DIR),
            unzip=True,
        )
        
        # Find the extracted folder (usually PlantVillage or plant_village)
        extracted = None
        for item in TEMP_DIR.iterdir():
            if item.is_dir():
                extracted = item
                break
        
        if extracted:
            if DATA_DIR.exists():
                shutil.rmtree(DATA_DIR)
            shutil.move(str(extracted), str(DATA_DIR))
            print(f"  Dataset extracted to: {DATA_DIR}")
            return True
        
    except Exception as e:
        print(f"  Kaggle failed: {e}")
    return False


def try_direct_download():
    """Download from alternative public sources."""
    import urllib.request
    
    # PlantVillage mirrors / alternative download URLs
    urls = [
        # GitHub hosted splits (smaller test download)
        "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip",
    ]
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TEMP_DIR / "plantvillage.zip"
    
    for url in urls:
        print(f"  Trying: {url[:80]}...")
        try:
            urllib.request.urlretrieve(url, str(zip_path))
            
            print("  Extracting...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(str(TEMP_DIR))
            
            # Find the color images directory
            for root, dirs, files in os.walk(str(TEMP_DIR)):
                if "color" in os.path.basename(root).lower() or any(
                    d.startswith("Apple") or d.startswith("Tomato") for d in dirs
                ):
                    if DATA_DIR.exists():
                        shutil.rmtree(DATA_DIR)
                    shutil.copytree(root, str(DATA_DIR))
                    print(f"  Dataset extracted to: {DATA_DIR}")
                    return True
            
            # If no color subfolder, look for class folders directly
            for item in TEMP_DIR.iterdir():
                if item.is_dir():
                    subdirs = list(item.iterdir())
                    for sub in subdirs:
                        if sub.is_dir():
                            inner_dirs = list(sub.iterdir())
                            for inner in inner_dirs:
                                if inner.is_dir() and (
                                    inner.name.startswith("Apple") or 
                                    inner.name.startswith("Tomato") or
                                    inner.name == "color"
                                ):
                                    target = inner if inner.name == "color" else inner.parent
                                    if DATA_DIR.exists():
                                        shutil.rmtree(DATA_DIR)
                                    shutil.copytree(str(target), str(DATA_DIR))
                                    print(f"  Dataset extracted to: {DATA_DIR}")
                                    return True
                            
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    return False


def verify_dataset():
    """Verify the dataset is properly structured."""
    if not DATA_DIR.exists():
        return False
    
    class_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if len(class_dirs) == 0:
        # Maybe there's a subdirectory
        for sub in DATA_DIR.iterdir():
            if sub.is_dir():
                inner = [d for d in sub.iterdir() if d.is_dir()]
                if len(inner) > 10:
                    # Move contents up
                    for item in sub.iterdir():
                        shutil.move(str(item), str(DATA_DIR / item.name))
                    sub.rmdir()
                    class_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
                    break
    
    total_images = 0
    for d in class_dirs:
        images = list(d.glob("*.jpg")) + list(d.glob("*.JPG")) + list(d.glob("*.png"))
        total_images += len(images)
    
    print(f"\n  Dataset verification:")
    print(f"    Location: {DATA_DIR}")
    print(f"    Classes: {len(class_dirs)}")
    print(f"    Total images: {total_images}")
    
    if len(class_dirs) >= 30 and total_images > 10000:
        print("    Status: ✅ Valid!")
        return True
    else:
        print("    Status: ❌ Incomplete")
        return False


def cleanup():
    """Remove temporary files."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  PlantVillage Dataset Downloader")
    print("=" * 60)
    
    # Check if already downloaded
    if DATA_DIR.exists() and verify_dataset():
        print("\n  Dataset already exists and is valid!")
        sys.exit(0)
    
    print()
    
    # Try different sources
    success = try_kaggle()
    if not success:
        success = try_direct_download()
    
    if success:
        verify_dataset()
        cleanup()
    else:
        print("\n  ❌ Automatic download failed.")
        print("  Please download manually:")
        print("    1. Go to https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("    2. Download the dataset")
        print(f"    3. Extract to: {DATA_DIR}")
        print("    4. Ensure folder structure: PlantVillage/<class_name>/<images>")
        cleanup()
        sys.exit(1)
