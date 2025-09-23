import os
import tensorrt as trt
from calibrator import Calibrator, CalibDataLoader

LOGGER = trt.Logger(trt.Logger.VERBOSE)

class_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush')

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.001,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
    "providers": ["CUDAExecutionProvider"]
}


def buildEngine(
    onnx_file, engine_file, mode, data_loader, calibration_table_path
):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    parser.parse_from_file(onnx_file)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 20))

    if mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    elif mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(data_loader, calibration_table_path)
    
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("EXPORT ENGINE FAILED!")

    with open(engine_file, "wb") as f:
        f.write(engine)


def main(mode):
    onnx_file = "/mnt/share_disk/bruce_trie/workspace/yolov8n.onnx"
    engine_file = f"/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/tensorrt_log/trt_yolov8/yolov8n_trt_{mode}.engine"
    calibration_cache = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/tensorrt_log/trt_yolov8/yolov8n_calib.cache"
    
    dataloader = CalibDataLoader(batch_size=1, calib_count=1024, info=info)

    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)

    print("Load ONNX file from:%s \nStart export, Please wait a moment..." % (onnx_file))
    buildEngine(onnx_file, engine_file, mode, dataloader, calibration_cache)
    print("Export ENGINE success, Save as: ", engine_file)


if __name__ == "__main__":
    # main("fp16")
    main("int8")
