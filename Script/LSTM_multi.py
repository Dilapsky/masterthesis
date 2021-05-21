import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
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
import torch.nn.functional as F
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
from collections import OrderedDict
import math
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
%matplotlib inline

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
set_seed(42)
cfg = {
    'format_version': 4,
    'data_path': "/kaggle/input/csvdata2",
    'model_params': {
        'model_architecture': 'LSTM',
        'model_name': "LSTM",
        'lr': 1e-3,
        'weight_path': "/kaggle/input/resnet34/resnet34.pth",

    },

    'train_params': {
        'max_num_steps':110 ,
        'checkpoint_every_n_steps': 25,
        'epoch':90
    }
}

total_df = pd.read_csv('/kaggle/input/csvdata2/LSTM_try.csv')
total_df.iloc[36000:]['Name'] = '000000'
test_df = total_df[total_df['Month']==11].copy()
for h in range(11):
    test_df = test_df.drop(index = 3455+h*3600)
set_diff_df = total_df.append(test_df)
set_diff_df = set_diff_df.drop_duplicates(keep=False)
class Mydataset(Dataset):
    def __init__(self,df_data):
        super().__init__()
        self.df=df_data
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idex):
        name=float(self.df['Name'].iloc[idex])
        month=float(self.df['Month'].iloc[idex])
        day=float(self.df['Day'].iloc[idex])
        hour=float(self.df['Hour'].iloc[idex])
        minute=float(self.df['Minute'].iloc[idex])
        speed=self.df['Speed'].iloc[idex]
        length=self.df['Length'].iloc[idex]
        tti=self.df['TTI'].iloc[idex]
        inputarrays = np.array([name,month,day,hour,minute,speed,length])
        label = np.array([tti])
        return inputarrays,label 
        
train_valid_dataset = Mydataset(df_data =set_diff_df)
test_dataset = Mydataset(df_data = test_df)
train_valid_loader = DataLoader(dataset=train_valid_dataset,batch_size=1,shuffle=False)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
class LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
model = LSTM()
print(model)
def forward(inputs, targets,model, criterion = nn.L1Loss()):
    preds = model(inputs.float())
    loss = criterion(preds.float(),targets.float())
    return loss, preds

def _val(loader, model,criterion = nn.L1Loss()):
    
    model.eval()
    valid_preds = np.zeros((len(loader.dataset), 1))
    avg_val_loss = 0.0
    for i, (i_batch, y_batch) in enumerate(loader):
        i_batch = i_batch
        y_batch = y_batch
        with torch.no_grad():
            y_pred = model(i_batch.float()).detach()
            avg_val_loss += criterion(y_pred.float(), y_batch.float()).item() / len(loader)
    return valid_preds, avg_val_loss

def _test(loader, model,criterion = nn.L1Loss()):
    model.eval()
    valid_preds = np.zeros((len(loader.dataset), 1))
    avg_val_loss = 0.0
    for i, (i_batch, y_batch) in enumerate(loader):

        with torch.no_grad():
            y_pred = model(i_batch.float()).detach()
            y_pred = y_pred.resize(1,1)
  
            avg_val_loss += criterion(y_pred.float(), y_batch.float()).item() / len(loader)
            valid_preds[i] = y_pred.cpu().numpy()
    return valid_preds, avg_val_loss
criterion = nn.L1Loss()
kf = KFold(n_splits=6, random_state=42, shuffle=True)
n = 0
for train_index, valid_index in kf.split(train_valid_dataset):
    n += 1
    print("K-Fold cross-validation, now is:", n,'/6')
    print("TRAIN:", train_index, "VALID:", valid_index)
    X_train, X_valid = total_df.iloc[train_index], total_df.iloc[valid_index]
    train_dataset = Mydataset(df_data = X_train)
    valid_dataset = Mydataset(df_data = X_valid)
    train_loader = DataLoader(dataset=train_dataset,batch_size=288,shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=60,shuffle=True)
    model = LSTM()
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor = 0.3,verbose=1,patience=8)
    optimizer.zero_grad()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
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
                inputs,labels = tr_it.next()
            except StopIteration:
                tr_it = iter(train_loader)
                inputs,labels = tr_it.next()
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(inputs.float())
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            losses_train.append(loss.item())
            progress_bar.set_description(f"train_loss: {loss.item()} train_loss(avg): {np.mean(losses_train)}")
            if i % cfg['train_params']['checkpoint_every_n_steps'] == 0:
                iterations.append(i)
                metrics.append(np.mean(losses_train))
                times.append((time.time()-start)/60)
    
        valid_preds, avg_val_loss = _val(valid_loader, model)
        scheduler.step(avg_val_loss)
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            print('Renew Validation Loss')
            torch.save(model.state_dict(), f'{model_name}_{n}_fold.pth')
        else:
            pass
        print('val_loss: ',avg_val_loss)
        print(f"Total training time is {(time.time()-start)/60} mins")
        print('Minimum Validation Loss is: ',min_val_loss)

Lastresult = np.zeros((test_dataset.__len__(),1))
for o in range(6):
    print('Predict test fold: ',o+1)
    modelx = LSTM()

    path = 'LSTM_'+str(o+1)+'_fold.pth'
    modelx.load_state_dict(torch.load(path))
    test_preds_foldx, avg_test_loss_foldx = _test(test_loader, modelx)
    Lastresult += (test_preds_foldx)/6
    
raw_df = pd.read_csv('/kaggle/input/csvdata2/Lasttotal_4th.csv')
previous_test_df = raw_df.iloc[3458:3602].copy()
previous_test_df['Time'] = list(raw_df['Time'].iloc[3457:3601])
result_df = previous_test_df.copy()
for b,w in enumerate(previous_test_df.columns.tolist()[1:]):
    result_df[w] = Lastresult[0+144*b:144*(b+1)]
result_df.to_csv('LSTM_result.csv',index = False)
target = np.array(test_df['TTI'])
target = np.reshape(target,(1584,1))
print('Average Loss of test file: ',np.sum(abs(target - Lastresult))/1584)

