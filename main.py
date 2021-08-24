import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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


c = []
lr = []

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for i in range(1000):
    if i % 10 == 0:
        learning_rate *= 0.995
        lr.append(learning_rate)
    coefs = theta(coefs)
    co = cost(coefs, X_train, y_train)[0]
    print(f"LR: {learning_rate} Lv: {i + 1} ==> Cost: {co}")
    c.append(co)

print("============================================================================")

print("b:", coefs[0][0])
print("W:", coefs[0][1:])

predict = []

for x in X_test:
    y_hat = h(coefs, x)
    if y_hat >= 0.5:
        predict.append(1)
    else:
        predict.append(0)

s = 0
for idx, i in enumerate(predict):
    if i == y_test[idx][0]:
        s += 1

print(f"Correct predictions {s} out of {len(predict)}.")

print("My Score:", s / len(predict))


plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.plot(c)
plt.show()

plt.xlabel("Iteration")
plt.ylabel("LR")
plt.plot(lr)
plt.show()