import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast__cancer.csv')

m = df.shape[0]
n = df.shape[1]

y = df.iloc[:, -1].values.reshape((m, 1))

normalized_data = (df - df.mean()) / df.std()

x = normalized_data.iloc[:, :-1]
x0 = np.ones((m, 1))
x = np.hstack((x0, x))

learning_rate = 0.0005

coefs = np.random.randn(1, n)

def h(coefs, x):
    HP = 1 / (1 + np.exp(- np.matmul(x, coefs.T)))
    return HP


def cost(coefs, x, y):
    diff = (y * (np.log(h(coefs, x)))) + ((1 - y) * (np.log(1 - h(coefs, x))))
    total = sum(diff)
    return total / (-m)


def theta(coefs):
    diff = h(coefs, x) - y
    mul = diff * x
    total = sum(mul)
    coefs = coefs - (learning_rate * total)

    return coefs