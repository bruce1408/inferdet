import onnx,json
from onnx import utils
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

additional_encodings = {
    "img_front": [
        {
            "bitwidth": 8,
            "dtype": "int",
            "scale": 0.01865844801068306,
            "offset": -114
        }
    ],
    "img_right": [
        {
            "bitwidth": 8,
            "dtype": "int",
            "scale": 0.01865844801068306,
            "offset": -114
        }
    ],
    "img_rear": [
        {
            "bitwidth": 8,
            "dtype": "int",
            "scale": 0.01865844801068306,
            "offset": -114
        }
    ],
    "img_left": [
        {
            "bitwidth": 8,
            "dtype": "int",
            "scale": 0.01865844801068306,
            "offset": -114
        }
    ]
}

input_names= [
    "/model.22/Reshape_output_0",
    "/model.22/Reshape_1_output_0",
    "/model.22/Reshape_2_output_0"
]

output_names = [
    "output0"
]

def get_tensor_info(tensor_name: str, is_float: bool = True) -> dict:
    
    if is_float:
        # 浮点数的默认编码
        bitwidth = 16
        dtype = "float"
    else:
        # 整数的默认编码，参考QNN常见设置
        bitwidth = 16
        dtype = "int"
    return {"bitwidth": bitwidth, "dtype": dtype}


def dump_model_info(
    onnx_model_path: str,
    output_json_path: str,
    additional_encodings: dict,
    quantify_parameters: bool = False,
    use_additional_encodings: bool = False,
    is_float: bool = True,
):
    print(f"正在加载模型生成量化encodings文件: {onnx_model_path}")
    model = onnx.load(onnx_model_path)
    graph = model.graph

    activation_encodings = {}
    param_encodings = {}
    
    if use_additional_encodings:
        print("正在应用额外的编码信息...")
        activation_encodings.update(additional_encodings)

    for node in graph.node:
        for output in node.output:
            if output not in activation_encodings:
                activation_encodings[output] = [get_tensor_info(output, is_float=is_float)]
    
    if quantify_parameters:
        param_encodings = {}
        for initializer in graph.initializer:
            param_name = initializer.name
            if param_name not in param_encodings:
                param_encodings[param_name] = [get_tensor_info(param_name, is_float=is_float)]
        model_info["param_encodings"] = param_encodings
        

    # 创建 JSON 数据
    model_info = {
        "activation_encodings": activation_encodings,
        "param_encodings": param_encodings
    }

    # 将 JSON 数据写入文件
    with open(output_json_path, 'w') as json_file:
        json.dump(model_info, json_file, indent=4)
        

onnx_model_path = cfg.DIPOORLET.yolov8_onnx_models
output_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/inferdet/tools/yolov8_extract.onnx"
output_json_dir = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/inferdet/tools"
output_json_path = f"{output_json_dir}/yolov8_overrides_mixed.json"

utils.extract_model(onnx_model_path, output_path, input_names, output_names)
dump_model_info(output_path, output_json_path, additional_encodings)