import torch
import os

abs_dir="D:/PythonProject/DeepLearning/Deep_learning/Models/"
def save_model(model,filename):
    print("model saved in {}".format(abs_dir+filename))
    torch.save(model,abs_dir+filename)

def check_model_exists(filepath):
    return os.path.exists(filepath)
def load_model(filepath,model):
    if not check_model_exists(filepath):
        return False
    mod=torch.load(filepath)
    print("loading model")
    model.load_state_dict(mod['model'])
    return True