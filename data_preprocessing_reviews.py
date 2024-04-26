import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#caller.set_index('key').join(other.set_index('key'))


random.seed(13)


def normalize(X_train, X_val, X_test):
    min_max_scaler = preprocessing.MinMaxScaler()

    # 计算训练集的统计信息并进行归一化
    X_train_normalized = min_max_scaler.fit_transform(X_train.values.astype(float))
    x_train = pd.DataFrame(X_train_normalized, columns=X_train.columns)

    # 使用训练集的统计信息对验证集进行归一化
    X_val_normalized = min_max_scaler.transform(X_val.values.astype(float))
    x_val = pd.DataFrame(X_val_normalized, columns=X_val.columns)

    # 使用训练集的统计信息对测试集进行归一化
    X_test_normalized = min_max_scaler.transform(X_test.values.astype(float))
    x_test = pd.DataFrame(X_test_normalized, columns=X_test.columns)

    return x_train, x_val, x_test



def split(dataset, val_frac=0.20, test_frac=0.10):
    X = dataset.loc[:, dataset.columns != 'price']
    X = X.loc[:, X.columns != 'id']
    X = X.loc[:, X.columns != 'host_id']
    X = X.loc[:, X.columns != 'Unnamed: 0']

    y = dataset['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(val_frac+test_frac), random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_frac/(val_frac+test_frac), random_state=1)

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":

    dataset = pd.read_csv('./Data/data_cleaned.csv')

    X_train, y_train, X_val, y_val, X_test, y_test = split(dataset)

    X_train, X_val, X_test = normalize(X_train, X_val, X_test)
    X_train.to_csv('./Data/data_cleaned_train_comments_X.csv', header=True, index=False)
    y_train.to_csv('./Data/data_cleaned_train_y.csv', header=True, index=False)

    X_val.to_csv('./Data/data_cleaned_val_comments_X.csv', header=True, index=False)
    y_val.to_csv('./Data/data_cleaned_val_y.csv', header=True, index=False)

    X_test.to_csv('./Data/data_cleaned_test_comments_X.csv', header=True, index=False)
    y_test.to_csv('./Data/data_cleaned_test_y.csv', header=True, index=False)


