import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

raw_total_df = pd.read_csv('/kaggle/input/csvdata2/Lasttotal_4th.csv')
total_df = raw_total_df.iloc[2:3602].copy()
total_df['Time'] = list(raw_total_df['Time'].iloc[1:3601])
road_id = total_df.columns[11]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.title('Time vs TTI in road No.'+road_id)
plt.ylabel('TTI')
plt.xlabel('Time per 10 min')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(total_df[road_id])
all_data = total_df[road_id].values.astype(float)
test_data_size = 144
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_window = 144
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
    
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
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
device = torch.device("cpu")
model = LSTM()
epochs = 30
criterion = nn.L1Loss()

valid_preds = np.zeros((576, 1))
avg_val_loss = 0.0
kf = KFold(n_splits=6, shuffle=False)
n = 0
for train_index, valid_index in kf.split(train_data):
    model = LSTM()
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor = 0.3,verbose=1,patience=3)
    n += 1
    print("K-Fold cross-validation, now is:", n,'/6')
    X_train, X_valid = train_data[train_index], train_data[valid_index]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_data_normalized = scaler.fit_transform(X_train .reshape(-1, 1))
    X_valid_data_normalized = scaler.fit_transform(X_valid .reshape(-1, 1))
    X_train_data_normalized = torch.FloatTensor(X_train_data_normalized).view(-1)
    X_valid_data_normalized = torch.FloatTensor(X_valid_data_normalized).view(-1)
    X_train_inout_seq = create_inout_sequences(X_train_data_normalized, train_window)
    X_valid_inout_seq = create_inout_sequences(X_valid_data_normalized, train_window)
    minloss = 100.0
    for i in range(epochs):
        
        valid_preds = np.zeros((576, 1))
        avg_val_loss = 0.0
        for seq, labels in X_train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
            model.to(device)
            seq = seq.to(device)
            labels = labels.to(device)
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        for seq, labels in X_valid_inout_seq:
            optimizer.zero_grad()
            with torch.no_grad():
                y_pred = model(seq).detach()
                avg_val_loss += criterion(y_pred.float(), labels.float()).item() / (len(X_valid_inout_seq)-test_data_size)
        if avg_val_loss < minloss:
            minloss = avg_val_loss
            print('Renew Validation Loss: ',avg_val_loss)
            torch.save(model.state_dict(), f'lstm_{n}_fold.pth')
        else:
            pass
        scheduler.step(avg_val_loss)

model.eval()
Lastresult = np.zeros((144, 1))
#------If you use predicted data to do prediction, use the following part
for o in range(6):
    print('Predict test fold: ',o+1)
    modelx = LSTM()
    modelx.to(device)
    
    path = 'lstm_'+str(o+1)+'_fold.pth'
    modelx.load_state_dict(torch.load(path))
    step_tensor = train_data_normalized[-train_window:]
    midresult = np.zeros((144, 1))
    if True:
        for s in range(144):
            optimizer.zero_grad()
            with torch.no_grad():
                y_pred = model(step_tensor).detach()
                midresult[s] = y_pred
                avg_val_loss += criterion(y_pred.float(), labels.float()).item() / 144
                step_tensor = torch.hstack((step_tensor,y_pred))[1:]
    actual_predictions = scaler.inverse_transform(midresult ).reshape(-1, 1)
    Lastresult += (actual_predictions)/6
print(Lastresult)
#------End of this part

#------If you use test data to do prediction, use the following part
model.eval()
Lastresult = np.zeros((144, 1))
for o in range(6):
    print('Predict test fold: ',o+1)
    modelx = LSTM()
    modelx.to(device)
    path = 'lstm_'+str(o+1)+'_fold.pth'
    modelx.load_state_dict(torch.load(path))
    test_inputs = train_data_normalized[-train_window:]
    test_data_normalized = scaler.fit_transform(test_data .reshape(-1, 1))
    test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)
    lasttensor = torch.hstack((train_data_normalized[-train_window:],test_data_normalized))
    test_inout_seq = create_inout_sequences(lasttensor, train_window)
    midresult = np.zeros((144, 1))
    for i in range(144):
        for s,(seq, labels) in enumerate(test_inout_seq):
            optimizer.zero_grad()
            with torch.no_grad():
                y_pred = model(seq).detach()
                midresult[s] = y_pred
                avg_val_loss += criterion(y_pred.float(), labels.float()).item() / 144
    actual_predictions = scaler.inverse_transform(midresult ).reshape(-1, 1)
    Lastresult += (actual_predictions)/6
print(Lastresult)
#-------End of this part

print('Total Loss of test file: ',np.sum(abs(test_data.reshape((144,1)) - Lastresult)))
x = np.arange(3457, 3601, 1)
plt.title('Time vs TTI')
plt.ylabel('TTI')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(total_df[road_id])
plt.plot(x,Lastresult)
plt.show()

plt.title('Time vs TTI')
plt.ylabel('TTI')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(total_df[road_id][-train_window:])
plt.plot(x,Lastresult)
plt.show()
np.savetxt('result'+road_id+'.txt',Lastresult)