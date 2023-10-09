import onnx
import numpy as np

# Load the ONNX model
model_path = "./graphs/simplified_model.onnx"
onnx_model = onnx.load(model_path)

# Initialize the total number of parameters
total_params = 0

# Iterate through model's initializer tensors
for initializer in onnx_model.graph.initializer:
    tensor = np.frombuffer(initializer.raw_data, dtype=np.float32)  # Assuming float32 dtype
    total_params += len(tensor)

print(f"Total number of model parameters: {total_params}")

