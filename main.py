import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


df = pd.read_csv('covid cases.csv')
X = np.array(df['Days']).reshape(-1, 1)  # features
y = np.array(df['cases']).reshape(-1, 1)  # target value
plt.plot(y, '-m')
polyfeat = PolynomialFeatures(degree=3)
X = polyfeat.fit_transform(X)
model = linear_model.LinearRegression()
model.fit(X, y)
accuracy = model.score(X, y)
print(round(accuracy*100, 3))
y1 = model.predict(X)
plt.plot(y1, '--b')
plt.show()

print("Training of model done... checking accuracy next....")


print("Done training and creating prediction model and ready for use.... ")
days = 7
print(round(int(model.predict(polyfeat.fit_transform([[707+days]])))/10000000, 2), 'cr')