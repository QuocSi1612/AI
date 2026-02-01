import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import append

data=pd.read_csv('data_classification.csv')

x_true=[]
y_true=[]
x_fasle=[]
y_false=[]

for item in  data.values:
    if item[2]==1:
        x_true.append(item[0])
        y_true.append(item[1])
    else:
        x_fasle.append((item[0]))
        y_false.append((item[1]))

plt.scatter(x_true,y_true,marker='o',color='red')
plt.scatter(x_fasle,y_false,marker='o',color='DarkBlue')

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def phanChia(p):
    if p>=0.5 :
        return 1
    return 0

def predict(feature, weights):
    z=np.dot(feature,weights)
    return sigmoid(z)

def cost_der_funtion(features,labels,weights):
    """
    :param features: 100x3
    :param labels: 100x1
    :param weights: 3x1
    :return: cost
    """
    n=len(labels)
    predictions=predict(features,weights)

    cost_class1=-labels*np.log(predictions)
    cost_class2=-(1-labels)*np.log(1-predictions)
    cost=cost_class1+cost_class2

    return cost.sum()/n

def update_weight(features,labels,weights,learning_rate):
    n=len(labels)

    predictions=predict(features,weights)
    gd=np.dot(features.T,(predictions-labels))

    gd=gd/n
    gd=gd*learning_rate
    weights=weights-gd
    return weights

def train(features,labels,weights,learning_rate,iter):
    cost_his=[]
    for i in range(iter):
        weights=update_weight(features,labels,weights,learning_rate)
        cost=cost_der_funtion(features, labels, weights)
        cost_his.append(cost)
    return weights,cost_his


features = data.iloc[:, :2].values  # Chọn cột đầu tiên và thứ hai làm đặc trưng
labels = data.iloc[:, 2].values  # Chọn cột thứ ba làm nhãn
features = np.hstack((np.ones((features.shape[0], 1)), features))  # Thêm cột bias (1)

# Khởi tạo trọng số ban đầu
weights = np.zeros(features.shape[1])  # Số chiều = số cột của features

# Thông số huấn luyện
learning_rate = 0.01
iterations = 1000

# Huấn luyện mô hình
weights, cost_history = train(features, labels, weights, learning_rate, iterations)

# Vẽ biểu đồ hàm cost

# Dự đoán trên toàn bộ dữ liệu
predictions = predict(features, weights)
predictions = [phanChia(p) for p in predictions]

# Tính độ chính xác
accuracy = sum(predictions == labels) / len(labels)
print(f"Accuracy: {accuracy * 100:.2f}%")# weight, bias, cost = train(x, y, 0.03, 0.0014, 0.001, 5000)
# print("Result: ")
# print(weight)
# print(bias)
# # print(cost)
# print("Giá trị dự đoán")
# print(predict(19, weight, bias))
# plt.show()