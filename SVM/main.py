import cv2
import numpy as np
from sklearn import svm
from sklearn import metrics

def setup_train_lables():
    lables = np.zeros((10,60000))
    for i in range(6000):
        for j in range(10):
            lables[j][i + (j * 6000)] = 1
    return lables

def setup_test_lables():
    lables = np.zeros((10,(10772 - 6000) * 10))
    for i in range(10772 - 6000):
        for j in range(10):
            lables[j][i + (j * (10772 - 6000))] = 1
    return lables

def load_train_data():
    data = np.zeros((60000,28 * 28))
    for i in range(6000):
        for j in range(10):
            data[i + j * 6000] = cv2.cvtColor(cv2.imread("D:\\Handwritten Digits\\dataset\\" + str(j) + "\\" + str(j)+"\\"+ str(i) + ".png"),cv2.COLOR_BGR2GRAY).flatten()
    return data

def load_test_data():
    data = np.zeros(((10772 - 6000) * 10,28 * 28))
    for i in range(10772 - 6000):
        for j in range(10):
            data[i + j * (10772 - 6000)] = cv2.cvtColor(cv2.imread("D:\\Handwritten Digits\\dataset\\" + str(j) + "\\" + str(j)+"\\"+ str(i + 6000) + ".png"),cv2.COLOR_BGR2GRAY).flatten()
    return data


train_data = load_train_data()
train_lables = setup_train_lables()
clfs = [svm.SVC(kernel="rbf",random_state=1, gamma=0.1, C=0.02)] * 10
test_data = load_test_data()
test_lables = setup_test_lables()
accuray = 0
for i in range(10):
    clfs[i].fit(train_data,train_lables[i])
    pred = clfs[i].predict(test_data)
    accuray += metrics.accuracy_score(pred,test_lables[i])
    
accuray /= 10
print(accuray)

