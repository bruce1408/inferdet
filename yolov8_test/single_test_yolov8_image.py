from algo import infer_yolov8
from loguru import logger
import numpy as np
from spectrautils.print_utils import *
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

class_names = cfg.DIPOORLET.COCO_labels

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "output_shape": [1, 84, 8400], # trt 需要 output_shape
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.5,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
    "providers": ["CPUExecutionProvider"]
}


def infer_and_show_results(mode="int8", img_path="", backend="onnx", raw_file_dir="", info=None):
    
    if backend == "tensorrt":
        with infer_yolov8(model_path, backend) as infer_instance:
            infer_instance.load_model(info)
            results, info = infer_instance.infer(img_path, info)
            infer_instance.show_results_single_img(img_path, results, class_names, f"{cfg.DIPOORLET.yolov8_outputs}/test_res_{backend}_724.jpg")
            print_colored_text(f"pic saved in :\n{cfg.DIPOORLET.yolov8_outputs}/test_res_{backend}_724.jpg", "green")
    
    elif backend == "onnx":
        infer_instance = infer_yolov8(model_path, "onnx")
        infer_instance.load_model(info)
        results, info = infer_instance.infer(img_path, info)
        infer_instance.show_results_single_img(img_path, results, info, f"{cfg.DIPOORLET.yolov8_outputs}/test_res_{BACKEND}.jpg")
        print_colored_text(f"pic saved in :\n{cfg.DIPOORLET.yolov8_outputs}/test_res_{BACKEND}.jpg", "green")
        
    elif backend == "qnn":
        infer_instance = infer_yolov8(model_path, "onnx")
        infer_instance.load_model(info)
        _, info = infer_instance.infer(img_path, info)

        # 从raw文件加载数据并进行后处理
        raw_file_path = f"{raw_file_dir}/Result_0/output0.raw"
        raw_data = np.fromfile(raw_file_path, dtype=np.float32)
        raw_data = raw_data.reshape(1, 84, 8400)
        outputs = [raw_data]
        results, info = infer_instance.postprocess(outputs, info)
        
        logger.info(f"Post-processed results from raw file: {results}")
        logger.info(f"results : {results}")
        infer_instance.show_results_single_img(img_path, results, info, f"{cfg.DIPOORLET.yolov8_outputs}/qnn_res_{mode}_556000.jpg")
        print_colored_text(f"pic saved in :\n{cfg.DIPOORLET.yolov8_outputs}/qnn_res_{mode}_556000.jpg", "green")
    
    
if __name__ == "__main__":
    
    # qnn and onnx are the same, trt use different model, support backend [qnn, onnx, tensorrt]
    BACKEND = "qnn"     
    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000000139.jpg"
    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000000285.jpg"
    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000001296.jpg"
    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000001675.jpg"
    # img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000000139.jpg"
    img_path = f"{cfg.SYSTEM.coco2017_val_path}/000000556000.jpg"
    
    if BACKEND == "tensorrt":
        model_path = f"{cfg.DIPOORLET.tensorrt_export_dir}/trt_yolov8/yolov8n_int8.engine"
    else:
        model_path = cfg.SYSTEM.yolov8_onnx_models
        if BACKEND == "qnn":
            QUANT_MODE = "mixed"
    
    # qnn infer result file dir
    raw_file_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_mixed_20250921_190437"

    infer_and_show_results(mode=QUANT_MODE, 
                           img_path=img_path, 
                           backend=BACKEND, 
                           raw_file_dir=raw_file_dir,
                           info=info)