import os
import math
from sklearn.naive_bayes import GaussianNB,CategoricalNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import  DecisionTreeClassifier
os.environ['OPENBLAS_NUM_THREADS'] = '1'

if __name__ == '__main__':
    dataset = globals()
    P = int(input())
    C = input().strip()
    L = int(input())

    split_at = int(len(dataset)*P/100)
    train_data = dataset[:split_at]
    test_data = dataset[split_at:]

    train_x = [row[:-1] for row in train_data]
    train_y = [row[-1] for row in train_data]

    test_x = [row[:-1] for row in test_data]
    test_y = [row[-1] for row in test_data]

    tree = DecisionTreeClassifier(criterion=C, max_depth=L,random_state=0)
    tree.fit(train_x, train_y)
    accuracy = tree.score(test_x,test_y)

    prediction_dict = {}

    for c in ['Perch', 'Roach', 'Bream']:
        tree.fit(train_x,[1 if y==c else 0 for y in train_y ])
        predictions = tree.predict(test_x)
        prediction_dict[c]=predictions

    count = 0
    for i,y in enumerate(test_y):
        good = True
        for c in ['Perch', 'Roach', 'Bream']:
          if y==c and prediction_dict[c][i]==1:continue
          if y!=c and prediction_dict[c][i]!=1:continue
          good = False
          break;
        if good:
            count += 1

    print(count/len(test_y))
