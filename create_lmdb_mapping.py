from tqdm import tqdm
import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True)
    args = parser.parse_args()

    lmdb_path = args.lmdb_path

    metadata_by_dataset = {}

    for dirpath, _, filenames in os.walk(lmdb_path):
        for filename in filenames:
            if filename.lower().endswith('.json') and filename.lower() != 'map.json':
                full_path = os.path.join(dirpath, filename)

                # split the path to get parent directory names
                parts = os.path.normpath(full_path).split(os.sep)

                try:
                    dataset_name = parts[-3]
                    subset_key = parts[-2]

                    with open(full_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    # initalize dataset key
                    if dataset_name not in metadata_by_dataset:
                        metadata_by_dataset[dataset_name] = {}

                    # loop through metadata to map image_names -> subset
                    for key in metadata['keys']:
                        metadata_by_dataset[dataset_name][key] = subset_key


                except (IndexError, json.JSONDecodeError) as e:
                    print(f"Skipping {full_path}: {e}")

    # save new metadata mapper
    for dataset_name, image_map in metadata_by_dataset.items():
        if 'VG' in dataset_name:
            output_path = os.path.join(lmdb_path, 'vg', dataset_name, 'map.json')
        else:
            output_path = os.path.join(lmdb_path, dataset_name, 'map.json')

        try:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(image_map, out_f, indent=2)

            print(f"Saved mapping to: {output_path}")
            
        except Exception as e:
            print(f"Failed to save {output_path}: {e}")
