import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

kf = KFold(random_state=123, shuffle=True)

import os
cpu = "48"
os.environ["OMP_NUM_THREADS"] = cpu # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = cpu # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = cpu # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = cpu # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = cpu # export NUMEXPR_NUM_THREADS=1
torch.set_num_threads(int(cpu))

def prep_nosubject(df):
    scs = {}
    for col in df.columns[3:]:
        sc = StandardScaler()
        df[col] = sc.fit_transform(df[col].values.reshape(-1,1))
        scs[col] = sc
    
    for sensor in df.columns[3:]:
        df[sensor+'_square'] = np.square(df[sensor])
        df[sensor+'_diff'] = df[sensor].diff()
        df.loc[df['step']==0, sensor+'_diff'] = 0.0
        df[sensor+'_mean5'] = df[sensor].rolling(5).mean().fillna(0)
        df[sensor+'_mean10'] = df[sensor].rolling(10).mean().fillna(0)
        df[sensor+'_mean20'] = df[sensor].rolling(20).mean().fillna(0)
    
    for col in df.columns[16:]:
        sc = StandardScaler()
        df[col] = sc.fit_transform(df[col].values.reshape(-1,1))
        scs[col] = sc
    
    return scs

class MyDataset(Dataset):
    def __init__(self, series, labels=None, roll=False):
        self.X = series.drop(columns=['sequence','subject','step']).values
        self.X = self.X.reshape(-1,60,series.shape[1]-3).copy()
        if labels is None:
            self.y = None
        else:
            self.y = labels['state'].values
        self.roll = roll
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ''' input tensor shape is N*L*C '''
        X = self.X[idx]
        toroll = np.random.choice(np.arange(-60,60))
        X = np.roll(X, toroll, axis=0)
        if self.y is None:
            y = -1
        else:
            y = self.y[idx]
        return (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

class MyModel(nn.Module):
    def __init__(self, input_feature, hidden_size, num_layers=1):
        super(MyModel, self).__init__()
        
        self.fc_pre = nn.Linear(input_feature, hidden_size)
        self.bn_pre = nn.BatchNorm1d(60)
        
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=0.5)
        ''' gru input is (N,L,H_in=H_hidden), output is (N,L,H_hidden), hidden is (num_layers, h_hidden)'''
        
        self.bn_post = nn.BatchNorm1d(hidden_size)
        self.fc_post = nn.Linear(hidden_size, 1)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)
    
        self.max = nn.MaxPool1d(60, padding=0)
        self.avg = nn.AvgPool1d(60, padding=0)

    def forward(self, input, hidden):
        ''' X is in the shape of (N,L,input_feature) '''
        output = self.dropout(F.relu(self.bn_pre(self.fc_pre(input))))
        output, hidden = self.gru(output, hidden)
        #output = self.dropout(F.relu(self.bn_post(output[:,-1,:])))

        #output = self.max(output.transpose(1,2)).squeeze(dim=2) 
        output = self.avg(output.transpose(1,2)).squeeze(dim=2) 
        output = self.dropout(F.relu(self.bn_post(output)))
 
        output = self.sigmoid(self.fc_post(output).squeeze(dim=1))
        return output
    
    def initHidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))

def evaluate(test_loader):
    model.eval()
    criteria = nn.BCELoss()
    with torch.no_grad():
        X, y = next(iter(test_loader))
        hidden = model.initHidden(len(X))
        output = model(X, hidden)
        score = roc_auc_score(y.detach().numpy(), output.detach().numpy())
        loss = criteria(output, y)
    return score, loss

def train(train_loader, test_loader, weight_decay, epoches, model_name, batch_size):

    criteria = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
    print(weight_decay)

    best_score = 0.0
    model_name = 'nn_model_'+str(weight_decay)+'_isplit_'+str(isplit)+'.pickle'
    for epoch in range(epoches):

        curr_loss = 0.0
        model.train()

        for batch, (X, y) in enumerate(train_loader):
            hidden = model.initHidden(len(X))
            output = model(X, hidden)
            loss = criteria(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_loss += loss.item() * len(y)

        curr_loss /= len(train_loader.dataset)
        score, test_loss = evaluate(test_loader)
        print(f"{weight_decay} {isplit} - {epoch}: {curr_loss}; test roc={score} test loss={test_loss}")

        if (score > best_score):
            torch.save({'epoch':epoch,'train_loss':curr_loss, 'test_loss':test_loss, 'score':score,
                    'model':model.state_dict()}, model_name)
            best_score = score
            print(f'saving model: {best_score}, {model_name}')


if (__name__ == "__main__"):
    train_series = pd.read_csv('train.csv')
    labels = pd.read_csv('train_labels.csv')
    test_series = pd.read_csv('test.csv')
    all_series = pd.concat([train_series, test_series], axis=0)
    scs = prep_nosubject(all_series)
    series = all_series.loc[all_series['sequence']<=25967]

    del train_series
    del test_series
    weight_decay = np.float32(input('enter weight decay:\n'))

    subjects = series['subject'].unique()

    for isplit, (train_index, test_index) in enumerate(kf.split(subjects)):
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]
        train_series = series.loc[series['subject'].isin(train_subjects)]
        test_series = series.loc[series['subject'].isin(test_subjects)]
        train_seqs = train_series['sequence'].unique()
        test_seqs = test_series['sequence'].unique()
        train_labels = labels.loc[labels['sequence'].isin(train_seqs)]
        test_labels = labels.loc[labels['sequence'].isin(test_seqs)]
        train_dataset = MyDataset(train_series, train_labels, roll=False) #doesn't have too much effect
        test_dataset = MyDataset(test_series, test_labels)

        model = MyModel(78, 32, num_layers=1)
        model_name = 'gru_model_'+str(weight_decay)+'_isplit_'+str(isplit)+'.pickle'

        batch_size = 128
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=len(test_dataset))
        train(train_loader, test_loader, weight_decay, 100, model_name, batch_size)

        batch_size = 512
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=len(test_dataset))
        train(train_loader, test_loader, weight_decay, 100, model_name, batch_size)
