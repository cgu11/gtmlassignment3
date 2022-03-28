from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import pandas as pd

def load_data(dataset="yeast", train_test_split_pct=0.1, split=True):
    if dataset == "yeast":
        data = pd.read_fwf('yeast.csv', header=None).dropna()
        x = data.iloc[:,1:-1].to_numpy()
        x_binary = x[:,4]
        x_binary[x_binary<1]=0
        y = data.iloc[:,-1].to_numpy()
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x[:,4] = x_binary

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    elif dataset == "WDBC":
        data = load_breast_cancer()
        x, y, labels, features = data.data, data.target, data.target_names, data.feature_names

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    if split:
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=train_test_split_pct, shuffle=True, random_state=555555, stratify=y)

        return train_x, test_x, train_y, test_y
    else:
        return x, y