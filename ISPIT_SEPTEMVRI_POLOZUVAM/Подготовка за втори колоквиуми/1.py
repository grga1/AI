import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from dataset_script import dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import math
if __name__ == '__main__':
    C = int(input())
    P = int(input())

    new_dataset = []
    for row in dataset:
        first = float(row[0])+float(row[-2])
        new_dataset.append([first]+row[1:-2]+[row[-1]])

    dataset = new_dataset
    good = [ row for row in dataset if row[-1]=='good']
    bad = [ row for row in dataset if row[-1]=='bad']

    P=P/100
    if C==0:
        train_data = good[:int(P*len(good))]+bad[:int(P*len(bad))]
        test_data = good[int(P*len(good)):]+bad[int(P*len(bad)):]
    elif C==1:
        train_data = good[:-int(P * len(good))] + bad[:-int(P * len(bad))]
        test_data = good[-int(P * len(good)):] + bad[-int(P * len(bad)):]

    test_x = [row[:-1] for row in test_data]
    test_y = [row[-1] for row in test_data]

    train_x = [row[:-1] for row in train_data]
    train_y = [row[-1] for row in train_data]

    scaler = MinMaxScaler(feature_range=(-1,1))
    classifier1 = GaussianNB()
    classifier1.fit(train_x, train_y)
    print(f'Tochnost so zbir na koloni: {classifier1.score(test_x, test_y)}')
    classifier2 = GaussianNB()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    classifier2.fit(train_x,train_y)
    classifier2.score(test_x,train_y)
