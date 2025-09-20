from algo import infer_yolov8
from loguru import logger
from spectrautils.print_utils import *
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

class_names = cfg.DIPOORLET.COCO_labels
model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/tensorrt_log/yolov8n_int8.engine"

backend = "tensorrt"

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "output_shape": [1, 84, 8400], # 1. 添加缺失的 output_shape
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.5,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
}

with infer_yolov8(model_path, backend) as infer_instance:

    infer_instance.load_model(info)


    img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000000139.jpg"
    img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000000285.jpg"
    img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000405691.jpg"
    results, info = infer_instance.infer(img_path, info)
    logger.info(f"results : {results}")
    logger.info(f"info : {info}")

    infer_instance.show_results_single_img(img_path, results, class_names, f"{cfg.DIPOORLET.yolov8_outputs}/test_res_trt_691.jpg")
    print_colored_text(f"pic saved in :\n{cfg.DIPOORLET.yolov8_outputs}/test_res_trt_691.jpg", "green")