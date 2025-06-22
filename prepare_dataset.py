import os
import argparse
import math
import sys
from pathlib import Path

def create_subsets(source_dir, dest_base_dir, num_subsets):
    """
    Splits image files from a source directory into a specified number of subsets
    using symbolic links to avoid data duplication.

    Args:
        source_dir (str): The directory containing the original image files.
        dest_base_dir (str): The directory where subset folders will be created.
        num_subsets (int): The number of even subsets to create.
    """
    source_path = Path(source_dir).resolve()
    dest_base_path = Path(dest_base_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory not found at '{source_path}'")
        sys.exit(1)

    # --- 1. Find all image files ---
    print(f"Scanning for images in '{source_path}'...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_paths = sorted([
        p for p in source_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ])

    total_images = len(image_paths)
    if total_images == 0:
        print("Error: No images found in the source directory.")
        sys.exit(1)
    
    print(f"Found {total_images} images.")

    if num_subsets > total_images:
        print(f"Warning: Number of subsets ({num_subsets}) is greater than the number of images ({total_images}).")
        print(f"Adjusting number of subsets to {total_images}.")
        num_subsets = total_images

    # --- 2. Calculate split points ---
    images_per_subset = math.ceil(total_images / num_subsets)
    print(f"Creating {num_subsets} subsets with approximately {images_per_subset} images each.")

    # --- 3. Create destination directories and symbolic links ---
    dest_base_path.mkdir(parents=True, exist_ok=True)
    
    start_index = 0
    for i in range(num_subsets):
        subset_dir_name = f"subset_{i:03d}"  # e.g., subset_000, subset_001
        subset_path = dest_base_path / subset_dir_name
        subset_path.mkdir(exist_ok=True)

        # Determine the slice of images for this subset
        end_index = min(start_index + images_per_subset, total_images)
        image_slice = image_paths[start_index:end_index]
        
        print(f"  Creating '{subset_dir_name}' with {len(image_slice)} images (indices {start_index}-{end_index-1})...")

        # Create a symbolic link for each image in the slice
        for image_file in image_slice:
            link_path = subset_path / image_file.name
            
            # os.symlink is cross-platform and works well here.
            # We must use absolute path for the source of the link.
            try:
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink() # Remove old link if it exists
                os.symlink(image_file, link_path)
            except OSError as e:
                print(f"    Error creating symlink for {image_file.name}: {e}")
                print("    You may need to run this script with administrator/root privileges on some systems (like Windows).")
                
        start_index = end_index

    print("\n---------------------------------")
    print("Subset creation complete.")
    print(f"Subsets are located in: '{dest_base_path.resolve()}'")
    print("---------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a directory of images into N even subsets using symbolic links.")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Path to the directory containing the original images.")
    parser.add_argument("--dest_dir", type=str, required=True,
                        help="Path to the base directory where subset folders will be created.")
    parser.add_argument("--num_subsets", type=int, required=True,
                        help="The number of subsets to create (e.g., 4 for 4 GPUs).")

    args = parser.parse_args()
    create_subsets(args.source_dir, args.dest_dir, args.num_subsets)