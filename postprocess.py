import lmdb
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
from tqdm import tqdm
import random
import json
from datetime import datetime

def dumps_npz(data_dict, compress=True):
    with io.BytesIO() as buffer:
        if compress:
            np.savez_compressed(buffer, **data_dict, allow_pickle=True)
        else:
            np.save(buffer, **data_dict, allow_pickle=True)
        return buffer.getvalue()

# load npz data from bytes
def load_npz_from_bytes(npz_bytes):
    with io.BytesIO(npz_bytes) as f:
        with np.load(f, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
        
def process_single_image(image_data, image_path, coverage_threshold, min_new_area_contribution_ratio):
    image_name, masks, generate_visualizations = image_data

    try:
        # load image
        image_path = os.path.join(image_path, image_name)
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_height, image_width = original_image.shape[:2]

        # iterative greedy mask selection
        selected_masks_original_format = []
        selected_mask_composite_ids = set() 

        covered_area_mask = np.zeros((image_height, image_width), dtype=bool)
        total_image_pixels = image_height * image_width
        covered_pixels = 0
        min_new_area_mask = total_image_pixels * min_new_area_contribution_ratio

        iteration_count = 0
        max_iterations = len(masks) * 2

        rebuilding_frames = []
        if generate_visualizations:
            rebuilding_frames.append((original_image.copy(), 0.0, len(masks)))

        while covered_pixels < total_image_pixels * coverage_threshold and iteration_count < max_iterations:
            best_mask_idx = -1
            max_new_area = 0
            
            for i, ann in enumerate(masks):
                # skip already selected mask
                if (image_name, ann['id']) in selected_mask_composite_ids:
                    continue

                current_mask_binary = ann['segmentation']
                newly_covered_pixels = np.sum(np.logical_and(current_mask_binary, ~covered_area_mask))

                if newly_covered_pixels > max_new_area:
                    max_new_area = newly_covered_pixels
                    best_mask_idx = i

            if best_mask_idx != -1 and max_new_area >= min_new_area_mask:
                selected_mask_ann = masks[best_mask_idx]
                selected_mask_binary = selected_mask_ann['segmentation']
                selected_mask_composite_ids.add((image_name, selected_mask_ann['id']))
                
                # update ann
                reconstructed_ann = {k: v for k, v in selected_mask_ann.items() if k != 'id'}
                reconstructed_ann['segmentation'] = selected_mask_binary
                selected_masks_original_format.append(reconstructed_ann)
                
                # update covered area
                covered_area_mask = np.logical_or(covered_area_mask, selected_mask_binary)
                covered_pixels = np.sum(covered_area_mask)
                
                # create next visualization frame
                if generate_visualizations:
                    current_coverage_percentage = (covered_pixels / total_image_pixels) * 100

                    # use last frame for tnew one
                    overlay_image = rebuilding_frames[-1][0].copy()
                    overlay_image[selected_mask_binary] = (
                        overlay_image[selected_mask_binary] * (1 - 0.4) + np.array([255, 0, 0], dtype=np.uint8) * 0.4
                    ).astype(np.uint8)

                    rebuilding_frames.append((overlay_image, current_coverage_percentage, len(masks)))
                
            else:
                break

            iteration_count += 1
        
        final_coverage_percent = (np.sum(covered_area_mask) / total_image_pixels) * 100

        return {
            'status': 'success',
            'image_name': image_name,
            'selected_mask_count': len(selected_masks_original_format),
            'final_coverage_percent': final_coverage_percent,
            'selected_masks_data_for_lmdb': selected_masks_original_format,
            'visual_frames': rebuilding_frames if generate_visualizations else None
        }
    
    except Exception as e:
        return {'status': f'error: {e}', 'image_name': image_name}

def save_rebuilding_frames_plot(image_name, rebuilding_frames, output_dir):
    if not rebuilding_frames:
        return
      
    num_frames_to_display = min(len(rebuilding_frames), 12)
    cols = 4
    rows = int(np.ceil(num_frames_to_display / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    _, _, num_original_masks = rebuilding_frames[-1]
    fig.suptitle(f"Progressive Rebuilding for: {image_name} \n Num Original Masks: {num_original_masks}", fontsize=16)

    for i in range(num_frames_to_display):
        ax = axes[i]
        frame_image, coverage_percent, _ = rebuilding_frames[i]
        ax.imshow(frame_image)
        if i == 0:
            ax.set_title(f"Original (0.0%)")
        else:
            ax.set_title(f"Iter {i} ({coverage_percent:.1f}%)")
        ax.axis('off')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_rebuilding_frames.png"))
    plt.close(fig) 
    print(f"saved rebuilding frames visualization for {image_name}")

def data_generator(lmdb_path, images_for_visualization):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for key, value in txn.cursor():
            image_name = key.decode('utf-8')

            try:
                npz_data = load_npz_from_bytes(value)
                masks = [{'id': idx, **ann} for idx, ann in enumerate(npz_data['masks'])]
                visualize_this_image = image_name in images_for_visualization
                
                yield (image_name, masks, visualize_this_image)

            except Exception as e:
                print(f'warning: failed to load data for {image_name}. skipping. error: {e}')
                continue
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path",type=str,required=True)
    parser.add_argument("--mask_path",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--coverage_threshold", type=float, default=1.0)
    parser.add_argument("--min_new_area_contribution_ratio", type=float, default=0.001)
    parser.add_argument("--num_images_to_visualize", type=int, default=21)
    args = parser.parse_args()

    # paths
    image_path = args.image_path
    lmdb_path = args.mask_path
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    # hyperparameters
    coverage_threshold = args.coverage_threshold
    min_new_area_contribution_ratio =args.min_new_area_contribution_ratio
    num_images_to_visualize = args.num_images_to_visualize

    # get lmdb keys
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            image_names = [key.decode('utf-8') for key, _ in tqdm(txn.cursor())]
        env.close()

    except lmdb.Error as e:
        print(f'fatal error: could not read keys from LMDB at {lmdb_path}. Error: {e}')

    total_images = len(image_names)
    num_images_to_visualize = min(num_images_to_visualize, total_images)
    images_for_visualization = set(random.sample(image_names, num_images_to_visualize))

    ### stream data and postprocess
    num_processes = cpu_count()
    print(f'starting postprocessing with {num_processes} CPUs')
    worker_func_partial = partial(
        process_single_image,
        image_path=image_path,
        coverage_threshold=coverage_threshold,
        min_new_area_contribution_ratio=min_new_area_contribution_ratio,
    )

    successful_count = 0
    failed_logs = []
    successful_keys = []

    lmdb_output_path = os.path.join(output_path, 'masks.lmdb')

    try:
        env_output = lmdb.open(lmdb_output_path, map_size=1099511627776)
        with env_output.begin(write=True) as txn_output, Pool(num_processes) as pool:
            data_iter = data_generator(lmdb_path, images_for_visualization)

            p_bar = tqdm(
                pool.imap_unordered(worker_func_partial, data_iter),
                total=total_images,
                desc='processing & writing images ...'
            )

            for result in p_bar:
                if result and result.get('status') == 'success':
                    successful_count += 1
                    image_name = result['image_name']
                    successful_keys.append(image_name)

                    # write data
                    if result['selected_masks_data_for_lmdb']:
                        npz_bytes = io.BytesIO()
                        np.savez_compressed(npz_bytes, masks=result['selected_masks_data_for_lmdb'])
                        txn_output.put(result['image_name'].encode('utf-8'), npz_bytes.getvalue())

                    # visualize eventually
                    if result.get('visual_frames'):
                        save_rebuilding_frames_plot(
                            result['image_name'],
                            result['visual_frames'],
                            output_path
                        )

                elif result:
                    failed_logs.append(f"  - {result.get('image_name')}: {result.get('status')}")
                else:
                    failed_logs.append("  - An unknown error occurred in a worker process.")

        env_output.close()

    except Exception as e:
        print(f"\nFATAL ERROR during main processing loop: {e}")
        if 'env_output' in locals() and env_output.is_open():
            env_output.close() # Attempt to clean up
        exit(1)

    print("\n--- Processing Summary ---")
    print(f"Total images attempted: {total_images}")
    print(f"Successfully processed and written: {successful_count}")
    print(f"Failed or skipped: {len(failed_logs)}")
    if failed_logs:
        print("Example failures (up to 5):")
        for failure_log in failed_logs[:5]:
            print(failure_log)

    # save metadata
    metadata = {
        "source_image_path": image_path,
        "source_mask_path": lmdb_path,
        "output_path": output_path,
        "generation_timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_parameters": {
            "coverage_threshold": coverage_threshold,
            "min_new_area_contribution_ratio": min_new_area_contribution_ratio,
        },
        "stats": {
            "total_source_images": total_images,
            "successfully_processed_count": successful_count,
            "failed_or_skipped_count": len(failed_logs),
        },
        # This is the most valuable part for future scripts
        "keys": sorted(successful_keys)
    }
    try:
        with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f'\nsaved metadata at {output_path}')
    except Exception as e:
        print(f'error saving metadata: {e}')

    print(f"processing complete. output saved in: {output_path}")