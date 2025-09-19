import os
from loguru import logger
from abc import abstractmethod, ABCMeta
import cv2
from backend import *


class infer(metaclass=ABCMeta):
    def __init__(self, model_path, backend):
        self.model_path = model_path
        if os.path.exists(self.model_path):
            logger.info(f"model_path : {self.model_path}")
        else:
            raise Exception(f"{self.model_path} does not exists.")

        self.backend = backend
        self.runtime = None
        if self.backend == "onnx":
            self.infer_model_func = infer_onnx
            self.load_model_func = load_onnx
        elif self.backend == "tensorrt":
            self.infer_model_func = infer_tensorrt
            self.load_model_func = load_tensorrt
        elif self.backend == "openvino":
            self.infer_model_func = infer_openvino
            self.load_model_func = load_openvino
        else:
            raise Exception(f"Not support {self.backend} backend.")

    def load_model(self, info):
        logger.info("Loading model...")
        # 重点：根据后端类型，处理不同的返回值
        if self.backend == "tensorrt":
            # 如果是 tensorrt，接收三个返回值，并将 runtime 保存起来
            self.model, info, self.runtime = self.load_model_func(self.model_path, info)
            
            # 新增：在加载模型后，只创建一次 context
            self.context = self.model.create_execution_context()
        else:
            # 如果是其他后端，保持不变
            self.model, info = self.load_model_func(self.model_path, info)
        logger.info("Loading model is finished.")
    
    def __enter__(self):
        # __enter__方法在进入 with 语句时被调用
        # 它需要返回一个对象，通常就是 self
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        # __exit__方法在退出 with 语句时被调用
        # 我们可以在这里执行明确的清理工作，或者什么都不做。
        # 在我们的场景下，只要 with 语句结束，对象的生命周期就结束了，
        # Python 的垃圾回收会自动安全地释放所有资源。
        logger.info("Exiting context and releasing resources...")
        self.context = None
        self.model = None
        self.runtime = None
    

    @abstractmethod
    def preprocess(self, img_path):
        pass

    def infer_model(self, inputs, info):
        # 重点：根据后端，传递不同的参数
        if self.backend == "tensorrt":
            # 将持久化的 context 传递给推理函数
            return self.infer_model_func(inputs, self.model, self.context, info)
        else:
            return self.infer_model_func(inputs, self.model, info)
        # return self.infer_model_func(inputs, self.model, info)

    @abstractmethod
    def postprocess(self, outputs, info):
        pass
    
    def infer(self, img_path, info):
        inputs, info = self.preprocess(img_path, info)
        outputs, info = self.infer_model(inputs, info)
        results, info = self.postprocess(outputs, info)
        return results, info

    def show_results_single_img(self, img_path, results, class_names, save_path):
        img = cv2.imread(img_path)
        for result in results:
            class_id, class_name, score, x1, y1, w, h = result
            (label_width, label_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
            cv2.putText(img, class_name, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(save_path, img)
