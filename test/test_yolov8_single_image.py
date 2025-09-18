from algo import infer_yolov8
from loguru import logger
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

class_names = cfg.DIPOORLET.COCO_labels
model_path = cfg.DIPOORLET.yolov8_onnx_models
backend = "onnx"

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.5,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
    "providers": ["CPUExecutionProvider"]
}

infer_instance = infer_yolov8(model_path, backend)

infer_instance.load_model(info)

img_path = f"{cfg.DIPOORLET.yolov8_val_path}/000000000139.jpg"
results, info = infer_instance.infer(img_path, info)
logger.info(f"results : {results}")
logger.info(f"info : {info}")

infer_instance.show_results_single_img(img_path, results, info, f"{cfg.DIPOORLET.yolov8_outputs}/test_res.jpg")
print(f"pic saved in  : {cfg.DIPOORLET.yolov8_outputs}/test_res.jpg")