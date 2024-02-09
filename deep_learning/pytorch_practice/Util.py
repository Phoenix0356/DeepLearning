import torch
import os

abs_dir= "/deep_learning/Models/"
class Util():
    @staticmethod
    def save_model(model, filename):
        print("model saved in {}".format(abs_dir + filename))
        torch.save(model, abs_dir + filename)
    @staticmethod
    def train(epochs, training_loader, model, optimizer, criterion):
        for epoch in range(epochs):
            for batch_size, (data, targets) in enumerate(training_loader):
                # data=data.reshape(data.shape[0],-1)
                # 前向传播
                score = model(data)
                loss = criterion(score, targets)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                # 更新参数
                optimizer.step()



def load_model(filepath,model):
    if not os.path.exists(filepath):
        return False
    mod=torch.load(filepath)
    print("loading model")
    model.load_state_dict(mod['model'])
    return True


def model_training(model_loaded,train_loader,model,criterion,optimizer,epochs,model_name):
    if not model_loaded:
        print("model not find, a model is under training")
        Util.train(epochs,train_loader,model,optimizer,criterion)

        trained_model={'model':model.state_dict(),'optimizer':optimizer.state_dict()}
        Util.save_model(trained_model,model_name)
    else:print("model is successfully loaded")

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.shape[0]
        print("Accuracy:{}%".format(num_correct / num_samples * 100))
    model.train()