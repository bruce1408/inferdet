import os
import json
from tqdm import tqdm
from algo import infer_yolov8
from pycocotools.coco import COCO
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()



# model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/tensorrt_log/yolov8n_int8_3.engine"
model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/tensorrt_log/yolov8n_fp16.engine"
val_path = cfg.SYSTEM.coco2017_val_path
annFile = cfg.SYSTEM.coco2017_gt_annFile

backend = "tensorrt"

class_names = cfg.DIPOORLET.COCO_labels

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "output_shape": [1, 84, 8400],
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.001,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
    "providers": ["CUDAExecutionProvider"]
}

# Load model
with infer_yolov8(model_path, backend) as infer_instance:
    infer_instance.load_model(info)


    with open(annFile, "r") as fp_gt:
        gt_data = json.load(fp_gt)

    detection_out_dict = {
    "images": gt_data["images"],
    "annotations": [],
    "categories": gt_data["categories"]
    }

    # Load all COCO val images
    coco = COCO(annFile)
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)


    ann_idx = 0
    # debug_imgs = 50
    # for img_idx in range(len(images)):
    for img_idx in tqdm(range(len(images)), ncols=100):

        file_name = images[img_idx]["file_name"]
        img_path = os.path.join(val_path, file_name)
        results, info = infer_instance.infer(img_path, info)
        
        for result in results:
            detection_out_dict['annotations'].append(
                {
                    "image_id": images[img_idx]["id"],
                    "bbox": [
                        result[3],
                        result[4],
                        result[5],
                        result[6]
                    ],
                    "category_id": gt_data["categories"][result[0]]["id"],
                    "id": ann_idx,
                    "score": result[2],
                    "area": result[5] * result[6]
                }
            )
            ann_idx += 1

    with open(f"{cfg.DIPOORLET.yolov8_outputs}/yolov8_trt_coco2017_fp16.json", "w") as fp_out:
        json.dump(detection_out_dict, fp_out, ensure_ascii=False, indent=4)
