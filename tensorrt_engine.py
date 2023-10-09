import torch.onnx
import torch
from collections import OrderedDict
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from torch.utils.data import DataLoader
import argparse
import yaml
import os
import cv2
from feeder.fashionMnist import Feeder
from torch.utils.data import DataLoader
from tqdm import tqdm
import ctypes
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
cuda.init()
feeder=Feeder(debug=False,state=1)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path',type=str, default='model.onnx')
    # parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--config', type=str, default='engine.yaml')
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--width', type=int, default=28)
    parser.add_argument('--save_engine', type=int, default=0)
    parser.add_argument('--save_engine_path', type=str, default='ctrgcn.engine')
    parser.add_argument('--load_engine_path', type=str, default=None)
    parser.add_argument('--is_test',type=int,default=0)
    parser.add_argument('--is_int8', type=int, default=0)
    parser.add_argument('--is_fp16', type=int, default=0)
    parser.add_argument('--test_batch', type=int, default=1)
    parser.add_argument('--fallback', type=int, default=1)
    parser.add_argument('--calBatchsize', type=int, default=1)
    parser.add_argument('--calibFile', type=str, default='calib.table')
    parser.add_argument('--soFile', type=str, default=None)
    return parser

def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    return opt

def read_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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


class CreateEngine:
    def __init__(self,args,logger):
        self.args=args
        self.logger=logger
        self.dataloader=DataLoader(feeder, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
        self.cfx = cuda.Device(0).make_context()
        pass
    def buildEngine(self):
        if self.args.load_engine_path is not None:
            with open(self.args.load_engine_path, 'rb') as f:
                serialized_engine = f.read()
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(serialized_engine)
                return
        if self.args.soFile:
            ctypes.cdll.LoadLibrary(self.args.soFile)
        builder = trt.Builder(self.logger)
        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
        major, minor, patch = trt.__version__.split('.')
        if int(major) >= 7:
            network_creation_flag |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_creation_flag)
        parser = trt.OnnxParser(network, self.logger)
        success = parser.parse_from_file(self.args.onnx_path)
        # print(success)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        if not success:
            exit(0)
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 28,28), (1, 3, 28,28), (1, 3, 28,28))
        config.add_optimization_profile(profile)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 31)  # 1<<20 is 1 MiB
        if self.args.is_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.args.is_int8:
            calib=Calibrator(self.args.calBatchsize,self.args,self.args.calibFile)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator=calib
        if self.args.fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        builder.build_engine(network, config)
        serialized_engine = builder.build_serialized_network(network, config)
        print('int8 process finished')
        if self.args.save_engine:
            with open(self.args.save_engine_path, 'wb') as f:
                f.write(serialized_engine)
        print('set runtime')
        print('set engine')

        if self.args.is_int8:
            calib.free()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
    def inference(self,data):

        context = self.engine.create_execution_context()
        context.set_binding_shape(0, data.shape)
        d_input = cuda.mem_alloc(data.nbytes)  # 分配输入的内存。
        output_shape = context.get_binding_shape(1)
        buffer = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(buffer.nbytes)  # 分配输出内存。
        cuda.memcpy_htod(d_input, data)
        bindings = [d_input, d_output]
        self.cfx.push()
        context.execute_v2(bindings)  # 可异步和同步
        cuda.memcpy_dtoh(buffer, d_output)
        output = buffer.reshape(output_shape)
        self.cfx.pop()
        return output

    def test(self):

        dataloader=DataLoader(feeder, batch_size=self.args.test_batch, shuffle=False, num_workers=0,
                   collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
        cnt=0
        correct=0
        process=tqdm(dataloader, ncols=40)
        for i,(data,label,idx) in enumerate(process):
            data=data.numpy()
            # print(data.shape)
            output=self.inference(data)

            output=np.argmax(output,axis=1)
            for k in range(0,self.args.test_batch):
                cnt+=1
                if label.numpy()[k]==output[k]:
                    correct+=1
            print(correct/cnt)
            return
        return

    def eval(self):
        dataloader=DataLoader(feeder, batch_size=self.args.test_batch, shuffle=False, num_workers=0,
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
            # print(data.shape)
            output = self.inference(data)

            output = np.argmax(output, axis=1)
            for k in range(0, self.args.test_batch):
                cnt += 1
                if label.numpy()[k] == output[k]:
                    correct += 1
        # 记录结束时间
        end_time = time.time()

        # 计算运行时间（以秒为单位）
        execution_time = end_time - start_time
        print('The accuracy is {}'.format(correct / cnt))
        print(f"代码执行时间：{execution_time:.4f} 秒")
        return

##读取数据


# 可以继承ILogger复写logger
# class MyLogger(trt.ILogger):
#     def __init__(self):
#        trt.ILogger.__init__(self)
#
#     def log(self, severity, msg):
#         pass # Your custom logging implementation here
#
# logger = MyLogger()



if __name__=='__main__':
    parser = parse_opt()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    createEngine=CreateEngine(arg,logger)
    createEngine.buildEngine()
    createEngine.eval()
        # inference=Inference()



