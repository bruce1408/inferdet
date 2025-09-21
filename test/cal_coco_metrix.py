from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

anno_json = "/mnt/share_disk/bruce_trie/misc_data_products/coco2017/images/annotations/instances_val2017.json"
# pred_json = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/dipoorlet_log/4_dipoorlet_models_yolov8/yolov8_onnx_coco2017_res.json"
# pred_json = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/inferdet/res1.json"
# pred_json = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/dipoorlet_log/4_dipoorlet_models_yolov8/yolov8_trt_coco2017.json"
pred_json = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/dipoorlet_log/4_dipoorlet_models_yolov8/yolov8_trt_coco2017_fp16.json"
anno = COCO(anno_json)
pred = COCO(pred_json)
eval = COCOeval(anno, pred, 'bbox')
eval.evaluate()
eval.accumulate()
eval.summarize()

print(eval.stats)
