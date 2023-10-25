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

current_dir=os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
input_size=28
sequence_length=28
num_layers=2
hidden_size=256
class_num=10
learning_rate=0.001
batch_size=64
epochs=2

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_class):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size*sequence_length,num_class)
    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out,_=self.rnn(x,h0)
        out=out.reshape(out.shape[0],-1)
        out=self.fc(out)
        return out


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: torch.squeeze(x, 1)
    ])
    #load dataset
    train_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=True,transform=transform,download=True)
    train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    test_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=False,transform=transform,download=True)
    test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)
    #load model
    model = RNN(input_size,hidden_size,num_layers,class_num)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)

    model_loaded=Util.load_model("D:\PythonProject\DeepLearning\Deep_learning\Models\Rnn_model.pth.tar",model)
    Util.model_training(model_loaded,train_loader,model,criterion=criterion,optimizer=optimizer,epochs=epochs,
                        model_name="Rnn_model.pth.tar")
    Util.check_accuracy(train_loader,model)
    Util.check_accuracy(test_loader,model)
