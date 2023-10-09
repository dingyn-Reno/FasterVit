import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class Model(nn.Module):
    def __init__(self,in_chans=1):
        super(Model,self).__init__()

        self.model = timm.create_model('resnet152',num_classes=10,in_chans=in_chans)


    def forward(self, input):
        x=self.model(input)

        return x

if __name__=='__main__':
    model=Model()
    input = torch.randn(1,1,256,256)
    from thop import profile
    flops, params = profile(model, inputs=(input,))
    print(flops, params)