import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import Util
import numpy as np

device=torch.device("cpu")

data_size=784
class_num=10
learning_rate=0.001
batch_size=64
epochs=1
class Linear_model(nn.Module):
    def __init__(self,input_size,class_nums):
        super(Linear_model,self).__init__()
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,class_nums)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
class CNN(nn.Module):
    def __init__(self,input_channal=1,num_class=10):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1=nn.Linear(16*7*7,num_class)
    def forward(self,x):

        x= F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)
        return x



if __name__=="__main__":
    train_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=True,transform=transforms.ToTensor(),download=True)
    train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    test_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=False,transform=transforms.ToTensor(),download=True)
    test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

    model = CNN()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)

    model_loaded=Util.load_model("D:\PythonProject\DeepLearning\Deep_learning\Models\Cnn_model.pth.tar",model)
    Util.model_training(model_loaded,train_loader,model,criterion,optimizer,epochs,"Cnn_model.pth.tar")

    Util.check_accuracy(train_loader,model)
    Util.check_accuracy(test_loader,model)




