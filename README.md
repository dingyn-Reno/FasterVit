## 简介

这是笔者开始学习模型推理之后所做的一个练手项目，基于FasterTransformer用于对ViT模型进行推理优化，本项目参考了FasterTransformer、Nvidia Hackerthon Cookbook和 TensorRT项目，最终在单卡4090上，实现了比直接推理快15倍以上，比FasterTransformer上的ViT模型快10%的int8模型推理速度。

## 准备

程序在单卡4090上进行，使用的数据为Fashion Mnist，基础精度为0.445。可选模型包括原始ViT（model/vit.py）和FasterTransformer构建的int8_ViT模型，（FasterTransformer/examples/pytorch/vit/ViT-quantization/vit_int8.py），完成简单训练后得到权重。

## 运行项目

```bash
# 转换onnx
python inference.py
# 化简计算图
polygraphy run evaluate-model your_model.onnx --onnxrt --trt
python onnx_optimizer.py
python modifiy_gs.py
# 配置config.yaml文件
# 运行tensorrt推理
python tensorrt_engine.py
```

## 将模型解析为onnx

代码inference.py首先将Pytorch版本的Vit转化为onnx模型，使用onnx初步推理时达到295.6190s。

## 计算图自动优化

我们测试的计算图的自动优化技术包括：PolyGraphy（常量折叠），onnx optimizer（删除不必要的节点和边），onnx-simplifier（计算图自动化简）。实验结果显示，onnx-simplifier和PolyGraphy用处较大，onnx optimizer几乎没有效果。最终计算图优化后速度达到290.6022s。

```python
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
```

```bash
polygraphy run evaluate-model your_model.onnx --onnxrt --trt
```

```python
import onnx
from onnxsim import simplify

model_path = "./graphs/folded.onnx"
onnx_model = onnx.load(model_path)

simplified_model, check = simplify(onnx_model)

simplified_model_path = "./graphs/simplified_model.onnx"
onnx.save(simplified_model, simplified_model_path)
```

## 计算图手工优化

计算图手工优化采用了onnx GraphSurgeon。本工程主要使用了对复杂SUT结构的消除，速度达到了289.8771s，实现了微小的提升。核心代码：

```python
for node in graph.nodes:
    if 'to_qkv/MatMul' in node.name:
        print(node.name)
        reshape_node=node.o(1).o()
        transpose_node=node.o(1).o().o().o()
        matmul_node=node.o(1).o().o().o().o()
        # matmul_node.inputs[1]=reshape_node.outputs[0]

        # print(transpose_node)
        trans_attrs={'perm': [0,2,3,1]}
        transpose_new=gs.Node("Transpose","transpose_new_{}".format(str(cnt)),inputs=[reshape_node.outputs[0]],outputs=[matmul_node.inputs[1]],attrs=trans_attrs)
        graph.nodes.append(transpose_new)
        transpose_node.outputs.clear()
        cnt+=1

# Remove the fake node from the graph completely
graph.cleanup()
```

## bulider和config的配置优化

在builder中设置标志位，但这个操作并未取得明显的性能提升。尺寸对齐同样，没有取得更好的结果。

## Plugin替换

ViT的plugin替换涉及到LayerNorm的替换工作。实现后推理速度达到286.7723s。我们使用的是tensorRT中提供的LayerNormKernel.cu内的核函数。原理上，主要是改进了reducesum的计算过程（通过共享内存以及对wrap的计算），还有对D（X）的计算采用了手写为E(X^2)-E(X)^2的方法。

核心代码：

```cpp
template <typename T, int32_t TPB, int32_t VPT, bool hasBias>
__global__ void skipln_vec(
    int32_t const ld, const T* input, const T* skip, T* output, const T* beta, const T* gamma, const T* bias)
{
    int32_t const idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    T inLocal[VPT];
    T skipLocal[VPT];
    T biasLocal[VPT];
    // T gammaLocal[VPT];
    copy<sizeof(T) * VPT>(&input[idx], inLocal);
    copy<sizeof(T) * VPT>(&skip[idx], skipLocal);
    copy<sizeof(T) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    T local = 0.f;
    T local2 = 0.f;

    const T rld = T(1) / T(ld);
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        inLocal[it] += skipLocal[it];
        if (hasBias)
            inLocal[it] += biasLocal[it];
        const T tmp = rld * inLocal[it];
        local += tmp;
        local2 += tmp * inLocal[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], biasLocal);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], skipLocal);

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    auto const sumKV = BlockReduce(tempStorage).Reduce(kvp<T>(local, local2), cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();
    ///*
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        inLocal[it] = skipLocal[it] * (inLocal[it] - mu) * rsigma + biasLocal[it];
    }
    /* */

    copy<sizeof(T) * VPT>(inLocal, &output[idx]);
}


```

## fp16/int8量化

量化是提高速度的核心操作，经fp16优化后达到236.6728s，int8量化后19.4612s。int8配置时如何校准激活层的操作可以见前文 https://dingyn-reno.github.io/2023/07/11/tensorint8/，目的自然是将分布散乱的较大的激活值舍弃掉。核心代码为：

```python
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batches,args,cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batches = batches
        self.cache_file = cache_file
        self.dataloader= DataLoader(feeder, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
        self.epoch=0
        self.arg=args
        self.device_input = [cuda.mem_alloc(trt.volume((16,512))* trt.int32.itemsize)]
    def free(self):
        for dinput in self.device_input:
            dinput.free()

    def get_batch_size(self):
        return self.batches

    def get_batch(self,names):
        # Assume self.batches is a generator that provides batch data.
        try:
            data = next(iter(self.dataloader))[0]
            print(self.epoch)
            self.epoch+=1
            # Assume that self.device_input is a device buffer allocated by the constructor.
            data = data.numpy()
            cuda.memcpy_htod(self.device_input[0], data)
            return self.device_input
        except:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None

    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
```

## 性能讨论

相比FasterTransformer，本方案使用了额外的计算图折叠、优化等技术，包括自动优化针对计算图进行onnx-GraphSurgeon手动优化，此外，经过测试，本方案使用的CUB V2版LayerNorm算子的速度优于FasterTransformer。

然而，作为一个成型的方案，本工程和FasterTransformer还有较大差距，作为对Transformer通用结构的大型推理优化方案，FasterTranformer主要面向对fp16模型的推理优化工作，而本方案即使在针对性优化的ViT上，也没能取得在fp16上取得更好的优化结果。同时，FasterTranformer对分布式推理的潜在支持以及对大batch的优化工作也是本方案没能实现的。
