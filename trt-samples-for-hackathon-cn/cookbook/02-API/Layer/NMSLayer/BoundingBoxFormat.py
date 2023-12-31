#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import tensorrt as trt
from cuda import cudart

data0 = np.load("NMSIOData.npz")["box"]
data1 = np.load("NMSIOData.npz")["score"]
nB = 1
nC = data0.shape[0]

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()  # need OptimizationProfile though in Static Shape mode
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, 4))
inputT1 = network.add_input("inputT1", trt.float32, (nB, nC, 1))
profile.set_shape(inputT0.name, [nB, nC, 4], [nB, nC, 4], [nB, nC, 4])
profile.set_shape(inputT1.name, [nB, nC, 1], [nB, nC, 1], [nB, nC, 1])
config.add_optimization_profile(profile)
#------------------------------------------------------------------------------- Network
maxOutput = network.add_constant([], np.int32(5000).reshape(-1))
nmsLayer = network.add_nms(inputT0, inputT1, maxOutput.get_output(0))
nmsLayer.bounding_box_format = trt.BoundingBoxFormat.CENTER_SIZES  # set box format
#------------------------------------------------------------------------------- Network
network.mark_output(nmsLayer.get_output(0))
network.mark_output(nmsLayer.get_output(1))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
dDDTensor = {}  # note whether a tensor is data-dependent
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
print("Before inference")
# context.get_tensor_shape(lTensorName[1]) here returns (-1, 3)
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])
    dDDTensor[lTensorName[i]] = (-1 in context.get_tensor_shape(lTensorName[i]))

bufferH = []
bufferH.append(data0)
bufferH.append(data1)
for i in range(nInput, nIO):
    if dDDTensor[lTensorName[i]]:  # deal with data-dependent output tensor
        # context.get_tensor_shape(lTensorName[i]) here returns (-1, 3), which can not be used as size of a buffer
        nMaxByteOutput = context.get_max_output_size(lTensorName[i])  # use get_max_output_size to get maximum of the output (padding to 2^k byte)
        dataType = engine.get_tensor_dtype(lTensorName[i])
        bufferH.append(np.empty([nMaxByteOutput // dataType.itemsize], dtype=trt.nptype(dataType)))
        #bufferH.append(np.empty([nB * nC * 3], dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))  # we can also calculate it manually, case by case
    else:  # normal output tensor
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))

bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

print("After inference")
# once after inference, context.get_tensor_shape(lTensorName[1]) will return real shape of output tensor, which can be used as the size of a buffer
for i in range(nIO):
    # context.get_tensor_shape(lTensorName[1]) here returns real shape of output tensor
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nInput, nIO):
    if dDDTensor[lTensorName[i]]:  # cut the data-dependent output buffer into real shape
        shapeReal = context.get_tensor_shape(lTensorName[i])
        bufferH[i] = bufferH[i].reshape(-1)[:np.prod(shapeReal)].reshape(shapeReal)  # take first np.prod(shapeReal) elements

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)
