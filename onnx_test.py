import onnxruntime as rt
from feeder.fashionMnist import Feeder
from torch.utils.data import DataLoader
import numpy as np
# 加载 ONNX 模型
onnx_model_path = "test_2.onnx"
session = rt.InferenceSession(onnx_model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

data=np.random.randn(8,50,16,64).astype(np.float32)

import time
# 记录开始时间
start_time = time.time()

input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: data})
print(output[0].shape)
output = np.argmax(output[0], axis=1)
end_time = time.time()

# 计算运行时间（以秒为单位）
execution_time = end_time - start_time
print(f"代码执行时间：{execution_time:.4f} 秒")