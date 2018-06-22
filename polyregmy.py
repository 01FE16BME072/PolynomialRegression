import numpy as np
from matplotlib import pyplot as plt

X = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
Y = np.array([4,5,6,8,11,15,20,30,50,100])

plt.scatter(X,Y,s = 50,color = 'red')

#Tunned this value after visualizing thr graph for around 10 times by Keerthi

b0 = 0
b1 = -10
b2 = -50
b3 = -10
b4 = 50
N = len(Y)
Learningrate = 0.001

a = X
b = np.array([0.01,0.04,0.09,0.16,0.25,0.36,0.49,0.64,0.81,1])
c = np.array([0.001,0.008,0.027,0.064,0.125,0.216,0.343,0.512,0.729,1])
d = np.array([0.00001,0.0016,0.0081,0.0256,0.0625,0.1296,0.2401,0.4096,0.6561,1])

for i in range(1000):
	Yguess = (b0 + (b1*a) + (b2*b) + (b3*c) + (b4*d))
	b0_gradient = (-2/N)*sum(Y-Yguess)
	b0 = b0 - (b0_gradient*Learningrate)
	b1_gradient = (-2/N)*sum(a*(Y-Yguess))
	b1 = b1 - (b1_gradient*Learningrate)
	b2_gradient = (-2/N)*sum(b*(Y-Yguess))
	b2 = b2 - (b2_gradient*Learningrate)
	b3_gradient = (-2/N)*sum(c*(Y-Yguess))
	b3 = b3 - (b3_gradient*Learningrate)
	b4_gradient = (-2/N)*sum(d*(Y-Yguess))
	b4 = b4 - (b4_gradient*Learningrate)

#print(b0,b1,b2,b3,b4)
Y_Predicted = (b0 + (b1*a) + (b2*b) + (b3*c) + (b4*d))

plt.plot(X,Y_Predicted,color = 'blue')
plt.show()
