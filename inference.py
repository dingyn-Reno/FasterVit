import torch.onnx
import torch
from model import vit
from collections import OrderedDict

is_compile=False
torch_model=vit.Model(classes=10)
# 转换的onnx格式的名称，文件后缀需为.onnx
onnx_file_name = "./model.onnx"
# 我们需要转换的模型，将torch_model设置为自己的模型
model = torch_model

# 如果使用编译模式，会进行以下操作
if is_compile==True:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model=torch.compile(model)

# 加载权重，将model.pth转换为自己的模型权重
# 如果模型的权重是使用多卡训练出来，我们需要去除权重中多的module. 具体操作可以见5.4节
weights=torch.load("/share/work_dir/mnist/runs-59-221250.pt")
weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])
keys = list(weights.keys())
# print(keys)
model.load_state_dict(weights)
# 导出模型前，必须调用model.eval()或者model.train(False)
model.eval()

batch_size = 1 # 随机的取值，当设置dynamic_axes后影响不大
dummy_input = torch.randn(batch_size, 3, 28,28) # dummy_input就是一个输入的实例，仅提供输入shape、type等信息

# 这组输入对应的模型输出
output = model(dummy_input)
# 导出模型
torch.onnx.export(model,        # 模型的名称
                  dummy_input,   # 一组实例化输入
                  onnx_file_name,   # 文件保存路径/名称
                  export_params=True,        #  如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                  opset_version=15,          # ONNX 算子集的版本，当前已更新到15
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names = ['input'],   # 输入模型的张量的名称
                  output_names = ['output'], # 输出模型的张量的名称
                  # dynamic_axes将batch_size的维度指定为动态，
                  # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})
