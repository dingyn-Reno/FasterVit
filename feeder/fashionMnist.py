import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
from einops import rearrange
#importing libraries for buliding neural network model.
import tensorflow as tf
import keras as k
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,BatchNormalization,Dropout,Flatten


import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

class Feeder(Dataset):
    def __init__(self,debug=False, state=0):
        self.debug=debug
        self.state=state
        self.load()

    def __len__(self):
        return len(self.df_train)

    def load(self):

        if self.state==0:
            self.df_train=pd.read_csv("./archive/fashion-mnist_train.csv")
        elif self.state==1:
            self.df_train = pd.read_csv("./archive/fashion-mnist_test.csv")

    def __getitem__(self, idx):

        y = self.df_train.iloc[idx, 0]
        x = self.df_train.iloc[idx, 1:]
        x = x.values
        x = x.reshape(28, 28)
        x = x[np.newaxis, :]
        x = np.repeat(x, 3, axis=0)
        return x.astype(float),y,idx

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod




if __name__=='__main__':
    a=Feeder()
    print(a[0][0].shape)


