
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

class Feeder(Dataset):
    def __init__(self,debug=False, state=0):
        self.debug=debug
        self.dataDownload()
        self.state=state
        self.load()
    def __len__(self):
        return len(self.data)

    def dataDownload(self):
        trans = transforms.ToTensor()
        self.mnist_train = torchvision.datasets.FashionMNIST(
            root="../data", train=True, transform=trans, download=True)
        self.mnist_test = torchvision.datasets.FashionMNIST(
            root="../data", train=False, transform=trans, download=True)

    def load(self):
        if self.state == 0:
            self.data = self.mnist_train
        elif self.state == 1:
            self.data = self.mnist_test
        else:
            print("error state")
            exit(0)

    def __getitem__(self, idx):
        data_numpy,label=self.data[idx]
        data_numpy=np.array(data_numpy)
        return data_numpy.astype(np.float32),label,idx

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__=='__main__':
    a=Feeder()
    print(a[0][0].shape)

