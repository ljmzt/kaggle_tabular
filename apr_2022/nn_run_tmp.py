import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
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

def prep_withsubject(df):
    scs = {}
    for col in df.columns[3:]:
        sc = StandardScaler()
        df[col] = sc.fit_transform(df[col].values.reshape(-1,1))
        scs[col] = sc
    
    for sensor in df.columns[3:]:
        df[sensor+'_square'] = np.square(df[sensor])
        df[sensor+'_diff'] = df[sensor].diff()
        df.loc[df['step']==0, sensor+'_diff'] = 0.0
        
        tmp = df.groupby(['subject','step']).agg({sensor:'mean'}).to_dict()[sensor]
        df[sensor+'_subject_mean'] = pd.Series(zip(df['subject'], df['step'])).map(tmp)
       
        tmp = df.groupby(['subject','step']).agg({sensor+'_square':'mean'}).to_dict()[sensor+'_square']
        df[sensor+'_square_subject_mean'] = pd.Series(zip(df['subject'], df['step'])).map(tmp)
       
        tmp = df.groupby(['subject','step']).agg({sensor+'_diff':'mean'}).to_dict()[sensor+'_diff']
        df[sensor+'_diff_subject_mean'] = pd.Series(zip(df['subject'], df['step'])).map(tmp)
    
    for col in df.columns[16:]:
        sc = StandardScaler()
        df[col] = sc.fit_transform(df[col].values.reshape(-1,1))
        scs[col] = sc
    
    return scs

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

    for col in df.columns[16:]:
        sc = StandardScaler()
        df[col] = sc.fit_transform(df[col].values.reshape(-1,1))
        scs[col] = sc

    return scs

class MyDataset(Dataset):
    def __init__(self, series, labels=None, roll=False):
        self.X = series.drop(columns=['sequence','subject','step']).values
        self.X = self.X.reshape(-1,60,series.shape[1]-3).transpose([0,2,1]).copy()
        if labels is None:
            self.y = None
        else:
            self.y = labels['state'].values
        self.roll = roll
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ''' input tensor shape is N*C*L '''
        X = self.X[idx]
        if self.roll:
            toroll = np.random.choice(np.arange(-60,60))
            X = np.roll(X, toroll, axis=1)
        if self.y is None:
            y = -1
        else:
            y = self.y[idx]
        return (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

class MyFCNModel(nn.Module):
    def __init__(self, input_channel):
        super(MyFCNModel, self).__init__()
        torch.manual_seed(123)

        pmode = 'circular'
        
        self.conv1d_1 = nn.Conv1d(input_channel, input_channel, 5, groups=input_channel, padding=2, padding_mode=pmode)
        self.bn_1 = nn.BatchNorm1d(input_channel)
        self.conv1d_2 = nn.Conv1d(input_channel, input_channel, 9, groups=input_channel, padding=4, padding_mode=pmode)
        self.bn_2 = nn.BatchNorm1d(input_channel)
        self.conv1d_3 = nn.Conv1d(input_channel, input_channel, 19, groups=input_channel, padding=9, padding_mode=pmode)
        self.bn_3 = nn.BatchNorm1d(input_channel)
        
        self.conv1d_4 = nn.Conv1d(3*input_channel, 32, 3, padding=1, padding_mode=pmode)
        self.bn_4 = nn.BatchNorm1d(32)
        
        self.conv1d_5 = nn.Conv1d(32, 1, 3, padding=1, padding_mode=pmode)
        self.bn_5 = nn.BatchNorm1d(1)
        
        self.avg = nn.AvgPool1d(60, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)
    
    def forward(self, X):
        ''' input shape (N,C,L) '''
        
        X1 = F.relu(self.bn_1(self.conv1d_1(X)))
        X2 = F.relu(self.bn_2(self.conv1d_2(X)))
        X3 = F.relu(self.bn_3(self.conv1d_3(X)))
        
        X = torch.cat([X1,X2,X3], dim=1)
        X = self.dropout(X)
        
        X = F.relu(self.bn_4(self.conv1d_4(X)))
        X = self.dropout(X)
        
        X = self.conv1d_5(X)
        
        X = self.avg(X).squeeze(dim=2)
        output = self.sigmoid(X.squeeze(dim=1))
        
        return output

def evaluate(test_loader):
    model.eval()
    criteria = nn.BCELoss()
    with torch.no_grad():
        X, y = next(iter(test_loader))
        output = model(X)
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
            output = model(X)
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
    batch_size = 512

    for isplit, (train_index, test_index) in enumerate(kf.split(subjects)):
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]
        train_series = series.loc[series['subject'].isin(train_subjects)]
        test_series = series.loc[series['subject'].isin(test_subjects)]
        train_seqs = train_series['sequence'].unique()
        test_seqs = test_series['sequence'].unique()
        train_labels = labels.loc[labels['sequence'].isin(train_seqs)]
        test_labels = labels.loc[labels['sequence'].isin(test_seqs)]
        train_dataset = MyDataset(train_series, train_labels, roll=True)
        test_dataset = MyDataset(test_series, test_labels)

        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=len(test_dataset))

        model = MyFCNModel(39)
        model_name = 'nn_model_'+str(weight_decay)+'_isplit_'+str(isplit)+'.pickle'
        train(train_loader, test_loader, weight_decay, 300, model_name, batch_size)
