import pandas as pd
import numpy as np
from scipy.signal import correlate
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def prep(X):
    mean = np.mean(X, axis=1)[:, np.newaxis]
    X = X - mean
    power = np.sum(np.power(X, 2), axis=1)
    power[power==0] = 1
    power = np.sqrt(power)[:,np.newaxis]
    X = X / power
    return X

def search(X0_norm, X1_norm, n_series=5000, n_win=10, template_len=15):
    ids = np.random.choice(range(len(X1_norm)), n_series)
    best1 = -999
    for template_len in [10,15,20]:
        for idx in ids:
            for i in np.random.choice(range(len(X1_norm[0])-template_len), n_win):
                template = X1_norm[idx, i:i+template_len].copy()
                template -= np.mean(template)
                template /= np.sqrt(np.sum(np.power(template,2)))
            
                X1_X1 = correlate(template[np.newaxis,:], X1_norm, mode='valid')[::-1]
                X1_X1_max = np.max(X1_X1, axis=1)
                X1_X1_max_median = np.median(X1_X1_max)
            
                X0_X0 = correlate(template[np.newaxis,:], X0_norm, mode='valid')[::-1]
                X0_X0_max = np.max(X0_X0, axis=1)
                X0_X0_max_median = np.median(X0_X0_max)
            
                if (X1_X1_max_median - X0_X0_max_median > best1):
                    best1 = X1_X1_max_median - X0_X0_max_median
                    best_template1 = template.copy()
                    print(template_len, idx, i, best1)

    return best_template1


if (__name__ == '__main__'):

    np.random.seed(123)

    labels = pd.read_csv('train_labels.csv',index_col='sequence')
    id0, id1 = labels.index[labels['state']==0], labels.index[labels['state']==1]

    sensor = input('input sensor:\n')
    series = pd.read_csv('train.csv',index_col=['sequence','subject','step'])[[sensor]]
    X0 = series.loc[id0].values.reshape(-1,60)
    X1 = series.loc[id1].values.reshape(-1,60)

    X1_norm = prep(X1)
    X0_norm = prep(X0)

    template1 = search(X0_norm, X1_norm, n_series=1000)
    df = pd.DataFrame(template1, columns=['template'])
    df.to_csv('template_'+sensor+'.csv', index=False)
    
    rr = np.zeros((25968,))
    rr0 = np.max(correlate(template1[np.newaxis,:],X0_norm,mode='valid')[::-1],axis=1)
    rr1 = np.max(correlate(template1[np.newaxis,:],X1_norm,mode='valid')[::-1],axis=1)

    rr[id0] = rr0
    rr[id1] = rr1
    
    df = pd.DataFrame(rr, columns=['corr'])
    df.to_csv('template_corr_'+sensor+'.csv', index=False)
    
