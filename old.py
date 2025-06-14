from segment_anything import sam_model_registry
import cv2
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
import time

if __name__ == '__main__':
    sam_ckpt_path="checkpoints/sam1/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')

    start_time = time.time()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7, 
        box_nms_thresh=0.7, 
        stability_score_thresh=0.85, 
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        #min_mask_region_area=100,
        points_per_batch=64,
    )

    image = cv2.imread('000001.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image2 = cv2.imread('000000000025.jpg')
    #image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    mask_data, mask_data_s, mask_data_m, mask_data_l = mask_generator.generate(image)
    print(f'total time {time.time() - start_time}')