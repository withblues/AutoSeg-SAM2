import os
import shutil

# Update these paths
symlink_dir = "videos/textvqa/subsets/subset_002"
destination_dir = "./big_image"

# List of large image symlink filenames (just the symlink names)
large_images = [
    "75cd66309f4183ae.jpg",
    "756562da1d31ec0e.jpg"
]

os.makedirs(destination_dir, exist_ok=True)

for fname in large_images:
    symlink_path = os.path.join(symlink_dir, fname)

    if not os.path.islink(symlink_path):
        print(f"Skipping {fname}: not a symlink")
        continue

    target_path = os.path.realpath(symlink_path)
    if not os.path.isfile(target_path):
        print(f"Skipping {fname}: target file does not exist or not a file")
        continue

    # Move target file to destination folder
    dest_path = os.path.join(destination_dir, os.path.basename(target_path))
    shutil.move(target_path, dest_path)
    print(f"Moved {target_path} to {dest_path}")

    # Delete the symlink
    os.unlink(symlink_path)
    print(f"Deleted symlink {symlink_path}")