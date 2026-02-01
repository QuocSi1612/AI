from email.header import Header
from sklearn import linear_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataFrame=pd.read_csv('Advertising Budget and Sales.csv')

x=dataFrame.values[:,2]
y=dataFrame.values[:,4]
# print(x)
# print(y)
# plt.scatter(x,y,marker='o',color='yellow')
# plt.show()

def predict(new_radio,weight,bias):
    return new_radio*weight+bias

def cost_funtion(X,y,weight,bias):
    n=len(X)
    sum_error=0
    for i in range(n):
        sum_error+= (y[i]-(weight*X[i]+bias))**2
    return sum_error/2*n

def update_weight(X,y,weight,bias,learning_rate):
    n=len(X)
    weight_temp=0.0
    bias_temp=0.0
    for i in range(n):
        weight_temp+=-2*X[i]*(y[i]-(X[i]*weight+bias))
        bias_temp+=-2*(y[i]-(X[i]*weight+bias))
    weight-=(weight_temp/n)*learning_rate
    bias-=(bias_temp/n)*learning_rate
    return weight,bias

def train(X,y,weight,bias,learning_rate,iter):
    cost_his=[]
    for i in range(iter):
        weight,bias=update_weight(X,y,weight,bias,learning_rate)
        cost=cost_funtion(X,y,weight,bias)
        cost_his.append(cost)
    return weight,bias,cost_his

weight,bias,cost=train(x,y,0.03,0.0014,0.001,5000)
print("Result: ")
print(weight)
print(bias)
# print(cost)
print("Giá trị dự đoán")
print(predict(19,weight,bias))



regr= linear_model.LinearRegression()
regr.fit(x.reshape(-1,1),y)

print("Result Sklearn:")
print(regr.coef_)
print(regr.intercept_)
duDoan=np.array([19])
print("Sklearn dự đoán: ")
print(duDoan,regr.predict(duDoan.reshape(-1,1)))

plt.plot(x,regr.predict(x.reshape(-1,1)))
plt.plot(x, x*weight+bias, color='red', label='Code chay')
plt.scatter(x,y,marker='o',color='yellow')
plt.legend()
plt.show()


