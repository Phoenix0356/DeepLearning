import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

input_size=28
sequence_length=28
num_layers=2
hidden_size=256
class_num=10
learning_rate=0.001
batch_size=64
epochs=1

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

train_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_set=datasets.MNIST(root='D:/MyDataSet/cnn',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

model = RNN(input_size,hidden_size,num_layers,class_num)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    for batch_size,(data,targets) in enumerate(train_loader):
        data=data.squeeze(1)
        # data=data.reshape(data.shape[0],-1)
        #前向传播
        score=model(data)
        loss=criterion(score,targets)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        #更新参数
        optimizer.step()

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
            x=x.squeeze(1)

            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.shape[0]
        print("Accuracy:{}%".format(num_correct/num_samples*100))
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
