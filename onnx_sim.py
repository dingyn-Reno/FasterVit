import onnx
from onnxsim import simplify

model_path = "./graphs/folded.onnx"
onnx_model = onnx.load(model_path)

simplified_model, check = simplify(onnx_model)

simplified_model_path = "./graphs/simplified_model.onnx"
onnx.save(simplified_model, simplified_model_path)
