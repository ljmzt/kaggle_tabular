import pandas as pd
import numpy as np
from scipy.signal import correlate
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

def prep(X):
    mean = np.mean(X, axis=1)[:, np.newaxis]
    X = X - mean
    power = np.sum(np.power(X, 2), axis=1)
    power[power==0] = 1
    power = np.sqrt(power)[:,np.newaxis]
    X = X / power
    return X

def search(X, y, n_series=500, n_win=10):
    best1 = -999
    for template_len in [3,5,10]:
        ids = np.random.choice(np.arange(len(y))[y==1], n_series)
        for k,idx in enumerate(ids):
            if (k % 100 == 1):
                print(template_len, k, idx)
            for i in np.random.choice(range(len(X[0])-template_len), n_win):
                template = X[idx, i:i+template_len].copy()
                template -= np.mean(template)
                power = np.sum(np.power(template,2))
                if (power == 0):
                    continue
                template /= np.sqrt(power)

                w = correlate(template[np.newaxis,:], X, mode='valid')[::-1]
                w = np.max(w, axis=1).reshape(-1,1)
                clf.fit(w,y)
                score = accuracy_score(y, clf.predict(w))

                if (score > best1):
                    best1 = score
                    best_template1 = template.copy()
                    print(template_len, idx, i, best1)

    return best_template1


if (__name__ == '__main__'):

    np.random.seed(123)

    y = pd.read_csv('train_labels.csv',index_col='sequence').values[:,0]

    sensor = input('sensor name:\n')
    series = pd.read_csv('train.csv',index_col=['sequence','subject','step'])[[sensor]]

    X = series.values.reshape(-1,60)
    X = prep(X)

    template1 = search(X, y, n_series=10)
    df = pd.DataFrame(template1, columns=['template'])
    df.to_csv('template_'+sensor+'.csv', index=False)

    rr = np.max(correlate(template1[np.newaxis,:],X,mode='valid')[::-1],axis=1)
    df = pd.DataFrame(rr, columns=['corr'])
    df.to_csv('template_corr_'+sensor+'.csv', index=False)
