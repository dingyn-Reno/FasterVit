import torch
from vit_pytorch import ViT
import torch.nn as nn



class Model(nn.Module):
    def __init__(self,classes=10):
        super(Model,self).__init__()

        self.model = ViT(
    image_size = 28,
    patch_size = 4,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
        )
        self.ReLu=torch.nn.ReLU()
        self.MLP=torch.nn.Linear(1000,classes)
        self.softmax=torch.nn.Softmax()

    def forward(self, input):
        with torch.no_grad():
            x=self.model(input)
        x=self.ReLu(x)
        x=self.MLP(x)
        x=self.softmax(x)

        return x

if __name__=='__main__':
    model=Model(classes=80)
    img = torch.randn(1, 3, 28, 28)
    from thop import profile

    flops, params = profile(model, inputs=(img,))
    print("估算的 FLOPS: {}, params: {}".format(flops,params))
    preds = model(img)  # (1, 1000)
    print(preds.shape)

    import time

    # 记录开始时间
    start_time = time.time()

    # 在这里放置你要测试运行时间的代码
    # 例如，计算一个大列表的和
    for i in range(0,1000):
        img = torch.randn(8,  3, 28, 28)
        output=model(img)
        print('epoch:{}'.format(i))


    # 记录结束时间
    end_time = time.time()

    # 计算运行时间（以秒为单位）
    execution_time = end_time - start_time

    print(f"代码执行时间：{execution_time:.4f} 秒")


