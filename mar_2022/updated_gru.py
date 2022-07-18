import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from itertools import product
from sklearn.metrics import mean_absolute_error as mae
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from sklearn.preprocessing import StandardScaler

look_back = 72
batch_size = 512
linear_node = 32
torch.set_num_threads(1)

def preprocess(dat):
    time_mapper = {}
    ii = 0
    for h in range(24):
        for mm in ['00','20','40']:
            hh = '{0:02d}'.format(h)
            time_mapper[hh+':'+mm] = ii
            ii += 1

    dat['unique'] = dat['x'].astype(str) + dat['y'].astype(str) + dat['direction']
    uniques = dat['unique'].unique()
    dat['day'] = pd.to_datetime(dat['time']).dt.weekday
    dat['time_stamp'] = dat['time'].apply(lambda x:time_mapper[x.split()[1][:5]])

    tmp = dat.groupby(['unique','day','time_stamp']).agg({'congestion':np.median})
    median_mapper = tmp.to_dict()['congestion']
    dat['median'] = dat.apply(lambda x: \
                              median_mapper[x['unique'],x['day'],x['time_stamp']], axis=1)
    dat['congestion-median'] = dat['congestion'] - dat['median']
    
    all_time = pd.DataFrame(pd.date_range('1991-04-01 00:00:00', '1991-09-30 11:40:00', freq='20Min'), columns=['time'])
    all_time['time'] = all_time['time'].astype(str)
    
    return uniques, median_mapper, time_mapper, all_time

def getseries(unique):
    df = dat.loc[dat['unique']==unique, ['time', 'congestion-median']]
    df = pd.merge(all_time, df, left_on='time', right_on='time', how='outer')
    df = df.set_index('time')
    df['congestion-median'] = df['congestion-median'].fillna(0)
    ss = StandardScaler()
    df['congestion-median-normalized'] = ss.fit_transform(df['congestion-median'].values.reshape(-1,1)).reshape(-1)
    return df, ss

def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)

def assemble(dat):
    train_loaders, test_loaders = [], []
    for period in test_periods_with_lookback:
        train = dat.loc[dat.index < period[0], 'congestion-median-normalized'].values
        test = dat.loc[(dat.index >= period[0]) & (dat.index <= period[1]), 'congestion-median-normalized'].values
        print(test[0])
        
        X, y = create_dataset(train, look_back=look_back)
        train_dataset = []
        for i in range(len(X)):
            train_dataset.append((torch.tensor(X[i].reshape(-1,1),dtype=torch.float32),
                                  torch.tensor(y[i].reshape(-1,),dtype=torch.float32)))
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, drop_last=False))
        
        X, y = create_dataset(test, look_back=look_back)
        test_dataset = []
        for i in range(len(X)):
            test_dataset.append((torch.tensor(X[i].reshape(-1,1),dtype=torch.float32),
                                 torch.tensor(y[i].reshape(-1,),dtype=torch.float32)))
        test_loaders.append(DataLoader(test_dataset, batch_size=batch_size, drop_last=False))
        
    return train_loaders, test_loaders

class MyModel(nn.Module):
    def __init__(self, input_feature, hidden_size, output_feature, num_layers=1):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_feature, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=0.2)
        ''' gru input is (N,L,H_in=H_hidden), output is (N,L,H_hidden), hidden is (num_layers, h_hidden)'''
        self.linear_out = nn.Linear(hidden_size, output_feature)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, input, hidden):
        ''' X is in the shape of (N,L,input_feature) '''
        output = F.relu(self.linear(input))
        output, hidden = self.gru(output, hidden)
        output = self.linear_out(F.relu(output))
        return output
    
    def initHidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))

def evaluate(test_loader):
    model.eval()
    with torch.no_grad():
        loss = 0
        n = 0
        for batch, (x, y) in enumerate(test_loader):
            h0 = model.initHidden(len(x))
            output = model.forward(x, h0)
            loss += criterion(output[:,-1,:],y).item() * len(x)
            n += len(x)
        loss /= n
    return loss

def train(n_epoches, train_loader, test_loader):
    optimizer = optim.Adam(model.parameters())
    
    best_test_loss = 100.0
    for epoch in range(n_epoches):
        
        curr_loss = 0.0
        model.train()
        
        n = 0
        for batch, (x, y) in enumerate(train_loader):
            h0 = model.initHidden(len(x))
            output = model.forward(x, h0)
            #print(output[-1,-1,:],y[-1])
            loss = criterion(output[:,-1,:], y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            curr_loss += loss.item()*len(x)
            n += len(x)
        
        curr_loss /= len(train_loader.dataset)
        test_loss = evaluate(test_loader)
        if (epoch % 20 == 0):  print(f'current {epoch} training loss={loss.item()} test loss = {test_loss}')
        if test_loss < best_test_loss:
            best_n_epoches = epoch + 1
            best_test_loss = test_loss
            print(f'updating best loss {epoch} training loss={loss.item()} test loss = {test_loss}')
            
        if epoch > best_n_epoches + 1:
            print('early stop')
            break
    return best_n_epoches

def retrain(n_epoches, train_loader):
    optimizer = optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(n_epoches):

        curr_loss = 0.0
        for batch, (x, y) in enumerate(train_loader):
            h0 = model.initHidden(len(x))
            output = model.forward(x, h0)
            loss = criterion(output[:,-1,:], y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_loss += loss.item()*len(x)

    curr_loss /= len(train_loader.dataset)
    return curr_loss

if (__name__ == "__main__"):
    dat = pd.read_csv('train.csv', index_col='row_id')
    uniques, median_mapper, time_mapper, all_time = preprocess(dat)
    criterion = nn.L1Loss()
    
    test_periods = [
    ['1991-09-16 12:00:00', '1991-09-16 24:00:00'],
    ['1991-09-23 12:00:00', '1991-09-23 24:00:00']]

    torch.manual_seed(123)
    i1 = int(input('starting unique id:\n'))
    i2 = int(input('starting unique id:\n'))
    for unique in uniques[i1:i2]:
        print(f"doing {unique}")
    
        df, ss = getseries(unique)
        print(ss.mean_, ss.scale_, df['congestion-median-normalized'].std())
    
        test_periods_with_lookback = []
        for period in test_periods:
            id1 = df.index.to_list().index(period[0])
            test_periods_with_lookback.append([df.index[id1-look_back], period[1]])
    
        model = MyModel(1, linear_node, 1, num_layers=3)
        train_loaders, test_loaders = assemble(df)
        best_n_epoches = train(200, train_loaders[0], test_loaders[0])
    
        model = MyModel(1, linear_node, 1, num_layers=3)
        print(f'refitting with {best_n_epoches}')
        curr_loss = retrain(best_n_epoches, train_loaders[1])
        print(f'curr_loss={curr_loss}')
    
        loss = torch.save({'loss': curr_loss,
                           'epoches': best_n_epoches,
                           'model': model.state_dict()}, 
                          'model_'+unique+'.pickle')
