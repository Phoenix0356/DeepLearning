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


device=torch.device("cpu")

data_size=784
class_num=10
learning_rate=0.001
batch_size=64
epochs=1

train_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

model = CNN().to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    for batch_size,(data,targets) in enumerate(train_loader):
        # data=data.reshape(data.shape[0],-1)
        #前向传播
        score=model(data)
        loss=criterion(score,targets)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        #更新参数
        optimizer.step()

trained_model={'model':model.state_dict(),'optimizer':optimizer.state_dict()}
Util.save_model(trained_model,"Cnn_model.pth.tar")

def check_accuracy(loader,model):
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            # x=x.reshape(x.shape[0],-1)
            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.shape[0]
        print("Accuracy:{}%".format(num_correct/num_samples*100))
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)




