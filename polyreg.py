import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataframe = pd.read_csv('Position_Salaries.csv')
#print(dataframe.head())
X = dataframe.iloc[:,1:2].values
Y = dataframe.iloc[:,2].values
#print(X)

reg = LinearRegression()
reg.fit(X,Y)

pol = PolynomialFeatures(degree = 4)
X_poly = pol.fit_transform(X)
reg_poly = LinearRegression()
reg_poly.fit(X_poly,Y)


plt.scatter(X,Y,s = 50,color = 'red')
plt.plot(X,reg.predict(X),color = 'blue')
plt.plot(X,reg_poly.predict(X_poly),color = 'green')
#plt.plot(X,reg_poly1.predict(X_poly1),color = 'yellow')
#plt.plot(X,reg_poly2.predict(X_poly2),color = 'black')
plt.show()

print(reg.predict(6.5))
print(reg_poly.predict(pol.fit_transform(6.5)))