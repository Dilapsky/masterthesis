#We recommend you to use computational stations with GPUs to run this code.
#Part of codes are refer from https://www.kaggle.com/huanvo/lyft-complete-train-and-prediction-pipeline by the author Huan Vo in Kaggle
from typing import Dict
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import os
import random
import time
import warnings
import math
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICE'] = '1,4,5'

#Fix the random seed to ensure the repeatability
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(42)

#Only adjust parameters here to change the configuation of deep learning
cfg = {
    
    'data_path': "./picture",
    'model_params': {
        'model_architecture': 'resnet34',
        'model_name': "model_resnet34_output",
        'lr': 1e-3,
    },
    'train_params': {
        'max_num_steps':160 ,
        'checkpoint_every_n_steps':40,
        'epoch':90
    }
}
#Padding the images from resolution 768*896 to 896*896 by padding two blanks with size 896*64 at the two sides of the images.
data_transforms = transforms.Compose([
        transforms.Pad(padding = (0,64), fill=0, padding_mode='constant'),
     #   transforms.Resize((896,896)),
        transforms.ToTensor(),
    ])

#Since we want to predict the TTI values 10 minutes after, so we move the 'Time' column one cell downwards.
total_df = pd.read_csv('./csvdata/Lasttotal_4th.csv')
train_valid_df = total_df.iloc[2:3458].copy()
train_valid_df['Time'] = list(total_df['Time'].iloc[1:3457])
test_df = total_df.iloc[3458:3602].copy()
test_df['Time'] = list(total_df['Time'].iloc[3457:3601])

#Change the format of time, the precision of the time should be 1 seconds
for u in range(len(train_valid_df)):
    tssl = train_valid_df['Time'].iloc[u]+':00'
    timearray = time.strptime(tssl,"%Y/%m/%d %H:%M:%S")
    otherStyleTime = time.strftime("%Y_%m_%d-%H-%M-%S",timearray)
    train_valid_df['Time'].iloc[u] = otherStyleTime+'.png'

for k in range(len(test_df)):
    tssl = test_df['Time'].iloc[k]+':00'
    timearray = time.strptime(tssl,"%Y/%m/%d %H:%M:%S")
    otherStyleTime = time.strftime("%Y_%m_%d-%H-%M-%S",timearray)
    test_df['Time'].iloc[k] = otherStyleTime+'.png'

#In pytorch, define the image loader
class Mydataset(Dataset):
    def __init__(self,df_data,data_dir='./picture',transform=data_transforms):
        super().__init__()
        self.df=df_data
        self.data_dir=data_dir
        self.transform=transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idex):
        img_name=self.df['Time'].iloc[idex]
        label=self.df.copy().drop(['Time'],axis = 1).iloc[idex]
        img_path=os.path.join(self.data_dir,img_name)
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        label = np.array(label)
        return img_tensor,label      
train_valid_dataset = Mydataset(df_data = train_valid_df,data_dir='./picture/',transform = data_transforms)
test_dataset = Mydataset(df_data = test_df,data_dir='./testpic/',transform = data_transforms)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

#In pytorch, define the model and forward propagation
class TransportModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=False, progress=True)
        self.backbone = backbone
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512
        self.head = nn.Sequential(
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )
        self.logit = nn.Linear(4096, out_features=11)
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = self.logit(x)
        return x

#Add more configuation like device(cpu or gpu), metrics, etc.
def forward(inputs, targets,model, device, criterion = nn.L1Loss()):
    inputs = inputs.to(device)
    targets = targets.to(device)
    preds = model(inputs)
    loss = criterion(preds.float(),targets.float())
    return loss, preds

#Add validation function for validation dataset
def _val(loader, model,device,criterion = nn.L1Loss()):
    model.eval()
    valid_preds = np.zeros((len(loader.dataset), 11))
    avg_val_loss = 0.0
    for i, (i_batch, y_batch) in enumerate(loader):
        i_batch = i_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_pred = model(i_batch).detach()
            avg_val_loss += criterion(y_pred.float(), y_batch.float()).item() / len(loader)   
    return valid_preds, avg_val_loss

#Add validation function for test dataset
def _test(loader, model,device,criterion = nn.L1Loss()):
    model.eval()
    valid_preds = np.zeros((len(loader.dataset), 11))
    avg_val_loss = 0.0
    for i, (i_batch, y_batch) in enumerate(loader):
        i_batch = i_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_pred = model(i_batch).detach()
            avg_val_loss += criterion(y_pred.float(), y_batch.float()).item() / len(loader)
            valid_preds[i] = y_pred.cpu().numpy()
    return valid_preds, avg_val_loss

#Use K-fold cross-validation, here I use K=6
kf = KFold(n_splits=6, random_state=43, shuffle=False)
n = 0
for train_index, valid_index in kf.split(train_valid_dataset):
    n += 1
    print("K-Fold cross-validation, now is:", n,'/6')
    X_train, X_valid = train_valid_df.iloc[train_index], train_valid_df.iloc[valid_index]
    train_dataset = Mydataset(df_data = X_train,data_dir='./picture/',transform = data_transforms)
    valid_dataset = Mydataset(df_data = X_valid,data_dir='./picture/',transform = data_transforms)
    train_loader = DataLoader(dataset=train_dataset,batch_size=18,shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=4,shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransportModel(cfg)
    #If you fine-tuning a pre-trained model, load the weight like this. Otherwise you don't use the following several rows. 
    weight_path = './model_resnet34_output_'+str(n)+'_fold.pth'
    if weight_path:
        print('load weight path: ',weight_path)
        model.load_state_dict(torch.load(weight_path))
    #End of weight load part
        
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    #The learning rate will reduce if the validation loss failed to decrease for 8 times.
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.3,verbose=1,patience=8)
    print(f'device {device}')
    tr_it = iter(train_loader)
    epoch = cfg["train_params"]["epoch"]
    min_val_loss = 100.0
    for k in range(epoch):
        progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
        num_iter = cfg["train_params"]["max_num_steps"]
        losses_train = []
        iterations = []
        metrics = []
        times = []
        model_name = cfg["model_params"]["model_name"]
        start = time.time()
        print('Begin Epoch: ',k)
        for i in progress_bar:
            try:
                images,labels = tr_it.next()
            except StopIteration:
                tr_it = iter(train_loader)
                images,labels = tr_it.next()
            model.train()
            torch.set_grad_enabled(True)
            loss, _ = forward(images,labels, model, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_train.append(loss.item())
            progress_bar.set_description(f"train_loss: {loss.item()} train_loss(avg): {np.mean(losses_train)}")
            if i % cfg['train_params']['checkpoint_every_n_steps'] == 0:
                iterations.append(i)
                metrics.append(np.mean(losses_train))
                times.append((time.time()-start)/60)
        valid_preds, avg_val_loss = _val(valid_loader, model,device)
        scheduler.step(avg_val_loss)
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            print('Renew Validation Loss')
            torch.save(model.state_dict(), f'{model_name}_{n}_fold.pth')
        else:
            pass
        print('val_loss: ',avg_val_loss)
    results = pd.DataFrame({'KFold': n,'Epoch': k,'iterations': iterations, 
    'metrics (avg)': metrics, 'elapsed_time (mins)': times})
    print(f"Total training time is {(time.time()-start)/60} mins")
    print('Minimum Validation Loss is: ',min_val_loss)
    print(results.head())

#Use your trained models to do prediction. Here I average the results generated by models from different folds, you can assign weights to different models.
Lastresult = np.zeros((test_dataset.__len__(), 11))
for o in range(6):
    print('Predict test fold: ',o+1)
    modelx = TransportModel(cfg)
    modelx.to(device)
    path = 'model_resnet34_output_'+str(o+1)+'_fold.pth'
    modelx.load_state_dict(torch.load(path))
    test_preds_foldx, avg_test_loss_foldx = _test(test_loader, modelx,device)
    Lastresult += (test_preds_foldx)/6
result_df = test_df.copy()
#Generate the final output and show the total loss of test file
for b,w in enumerate(test_df.columns.tolist()[1:]):
    result_df[w] = Lastresult[:,b]  
result_df.to_csv('CNN_resnet34_6fold.csv')
target = np.array(test_df.copy().drop(['Time'],axis = 1))
print('Total Loss of test file: ',np.sum(abs(target - Lastresult)))
