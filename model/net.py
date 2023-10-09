import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,in_chans=1,kernel_size=5,padding=2,stride=2):
        super(Model,self).__init__()
        self.f1=nn.Conv2d(in_chans, 6, kernel_size=kernel_size,padding=padding)
        self.f2=nn.Conv2d(6, 16, kernel_size=kernel_size)
        self.l1=nn.Linear(1600, 120)
        self.l2= nn.Linear(120, 84)
        self.l3= nn.Linear(84, 10)
        self.GAP=nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, input):
        x=self.f1(input)
        x=nn.Sigmoid()(x)
        x=self.f2(x)
        x=nn.Sigmoid()(x)
        x=self.GAP(x)
        x=nn.Flatten()(x)
        x=self.l1(x)
        x=nn.Sigmoid()(x)
        x=self.l2(x)
        x=nn.Sigmoid()(x)
        x=self.l3(x)

        return x

if __name__=='__main__':
    model=Model()
    input = torch.randn(1,1,28,28)
    from thop import profile
    flops, params = profile(model, inputs=(input,))
    print(flops, params)