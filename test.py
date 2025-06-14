from segment_anything import sam_model_registry
import cv2
from segment_anything.automatic_batched_mask_generator import SamAutomaticMaskGenerator


if __name__ == '__main__':
    sam_ckpt_path="checkpoints/sam1/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.7, 
        box_nms_thresh=0.7, 
        stability_score_thresh=0.85, 
        crop_n_layers=0,
        crop_n_points_downscale_factor=0,
        min_mask_region_area=100,
    )

    image = cv2.imread('000000000009.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('000000000025.jpg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    final_results_per_image = mask_generator.generate([image, image2])

    print(len(final_results_per_image))