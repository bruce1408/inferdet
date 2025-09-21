from algo import infer_yolov8
from loguru import logger
import numpy as np
from spectrautils.print_utils import *
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

class_names = cfg.DIPOORLET.COCO_labels
# log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_fp16_20250921_010631"
log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_int8_1000_20250920_215500"

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

def main(info):
    infer_instance = infer_yolov8(model_path, backend)
    infer_instance.load_model(info)


    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000000139.jpg"
    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000000285.jpg"
    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000001296.jpg"
    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000001675.jpg"
    img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000556000.jpg"
    results, info = infer_instance.infer(img_path, info)

    raw_file_path = f"{log_dir}/Result_0/output0.raw"
    raw_data = np.fromfile(raw_file_path, dtype=np.float32)
    raw_data = raw_data.reshape(1, 84, 8400)
    outputs = []
    outputs.append(raw_data)
    results, info = infer_instance.postprocess(outputs, info)

    mode="int8"

    logger.info(f"results : {results}")
    infer_instance.show_results_single_img(img_path, results, info, f"{cfg.DIPOORLET.yolov8_outputs}/qnn_res_{mode}_556000.jpg")
    print_colored_text(f"pic saved in :\n{cfg.DIPOORLET.yolov8_outputs}/qnn_res_{mode}_556000.jpg", "green")
    
    
if __name__ == "__main__":
    main(info)