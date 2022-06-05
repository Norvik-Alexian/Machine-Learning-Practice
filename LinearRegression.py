import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file = np.genfromtxt('./dataset/BostonHousing.csv', delimiter=',', skip_header=1)

age = file[:, 6]
tax = file[:, 9]

y = tax[:len(tax) // 2]
x = age[:len(age) // 2]

x_test = tax[len(tax) // 2:]
y_test = age[len(age) // 2:]

# plt.scatter(x, y)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

model = LinearRegression()
x = x.reshape(-1, 1)

model.fit(x, y)

a = model.coef_
b = model.intercept_

# print(f'{a}\n{b}')

'''
formula:
    minerror = sumn(y_predict - y) / count(x_test)
'''

sumn = 0
for i in range(len(y_test)):
    sumn += model.predict([[x_test[i]]]) - y_test[i]

error = sumn / len(x_test)

print(error)