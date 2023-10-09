import onnx
from onnxoptimizer import optimize

model_path = "./graphs/folded.onnx"
onnx_model = onnx.load(model_path)


# 删除不必要的节点和边
# optimized_model = optimize(onnx_model, ["eliminate_identity"])

# 权重融合
optimized_model = optimize(onnx_model, ["fuse_add_bias_into_conv"])

optimized_model_path = "./graphs/optimized_fuse_model.onnx"
onnx.save(optimized_model, optimized_model_path)

