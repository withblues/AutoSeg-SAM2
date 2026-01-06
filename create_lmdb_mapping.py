from tqdm import tqdm
import os
import json
import argparse

def old_merging(lmdb_path):
    metadata_by_dataset = {}

    for dirpath, _, filenames in os.walk(lmdb_path):
        for filename in filenames:
            if filename.lower().endswith('.json'):
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

def merge_dataset_jsons(lmdb_path):
    # Iterate top-level datasets
    for dataset_name in os.listdir(lmdb_path):
        dataset_path = os.path.join(lmdb_path, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        merged_map = {}

        if dataset_name == 'vg':
            # VG is special: iterate VG variants under vg/
            vg_path = dataset_path
            for vg_variant in os.listdir(vg_path):
                variant_path = os.path.join(vg_path, vg_variant)
                if not os.path.isdir(variant_path):
                    continue

                # iterate subsets
                for subset_name in os.listdir(variant_path):
                    subset_path = os.path.join(variant_path, subset_name)
                    json_path = os.path.join(subset_path, 'map.json')
                    if not os.path.exists(json_path):
                        continue
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            subset_map = json.load(f)
                        merged_map.update(subset_map)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON: {json_path}")

                # Save merged map for this variant
                output_path = os.path.join(vg_path, vg_variant, 'mapper.json')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_map, f, indent=2)
                print(f"Saved merged map to: {output_path}")

        else:
            # Other datasets: iterate subset folders directly
            for subset_name in os.listdir(dataset_path):
                subset_path = os.path.join(dataset_path, subset_name)
                json_path = os.path.join(subset_path, 'map.json')
                if not os.path.exists(json_path):
                    continue
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        subset_map = json.load(f)
                    merged_map.update(subset_map)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {json_path}")

            # Save merged map
            output_path = os.path.join(dataset_path, 'mapper.json')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_map, f, indent=2)
            print(f"Saved merged map to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True)
    args = parser.parse_args()

    lmdb_path = args.lmdb_path

    merge_dataset_jsons(lmdb_path)


