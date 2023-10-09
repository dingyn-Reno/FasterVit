import onnxruntime as rt
from feeder.fashionMnist import Feeder
from torch.utils.data import DataLoader

# 加载 ONNX 模型
onnx_model_path = "model.onnx"
session = rt.InferenceSession(onnx_model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

import numpy as np
feeder=Feeder(debug=False,state=1)
from tqdm import tqdm
dataloader=DataLoader(feeder, batch_size=1, shuffle=False, num_workers=0,
                   collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
cnt=0

correct=0
import time
# 记录开始时间
start_time = time.time()

# 在这里放置你要测试运行时间的代码
# 例如，计算一个大列表的和
process=tqdm(dataloader, ncols=40)
for i,(data,label,idx) in enumerate(process):
    data = np.array(data,dtype=np.float32)
    print(data)
    # print(data.shape)
    # output = self.inference(data)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: data})
    print(output[0].shape)
    output = np.argmax(output[0], axis=1)
    for k in range(0, 1):
        cnt += 1
        if label.numpy()[k] == output[k]:
            correct += 1
        # 记录结束时间

end_time = time.time()

        # 计算运行时间（以秒为单位）
execution_time = end_time - start_time
print('The accuracy is {}'.format(correct / cnt))
print(f"代码执行时间：{execution_time:.4f} 秒")