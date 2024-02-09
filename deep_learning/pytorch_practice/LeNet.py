import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import Util
import os
import numpy as np

in_channel=1
batch_size=64
learning_rate=0.01
epochs=2
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.relu=nn.ReLU()
        self.pool=nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),stride=(1,1))
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),stride=(1,1))
        self.conv3=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5),stride=(1,1))
        self.linear1=nn.Linear(120,84)
        self.linear2=nn.Linear(84,10)
    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x=x.reshape(x.shape[0],-1)
        x=self.relu(self.linear1(x))
        x=self.linear2(x)
        return x





if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])
    #load dataset
    train_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=True,transform=transform,download=True)
    train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    test_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=False,transform=transform,download=True)
    test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)
    #load model
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_loaded=Util.load_model("D:\PythonProject\DeepLearning\Deep_learning\Models\LeNet_model.pth.tar",model)

    Util.model_training(model_loaded,train_loader,model,criterion=criterion,optimizer=optimizer,epochs=epochs,
                        model_name="LeNet_model.pth.tar")

    Util.check_accuracy(train_loader,model)
    Util.check_accuracy(test_loader,model)