import matplotlib.pyplot as plt
import numpy as np




def Sa(x):
    return np.sin(x) / x

T_0 = 2 * np.pi
w_0 = 2 * np.pi / T_0
tau = T_0 / 16
A = 1
width = 10

'''
fig,axes=plt.subplots(2,2)
tau = T_0 / 2
w = [n * w_0 for n in range(-width, width)]
y = [A * tau * w_0 * Sa((n * w_0 * tau) / 2) for n in range(-width, width)]
axes[0][0].stem(w, y)
axes[0][0].set_title('r = T0 / 2')

tau = T_0 / 4
w = [n * w_0 for n in range(-width, width)]
y = [A * tau * w_0 * Sa((n * w_0 * tau) / 2) for n in range(-width, width)]
axes[0][1].stem(w, y)
axes[0][1].set_title('r = T0 / 4')

tau = T_0 / 8
w = [n * w_0 for n in range(-width, width)]
y = [A * tau * w_0 * Sa((n * w_0 * tau) / 2) for n in range(-width, width)]
axes[1][0].stem(w, y)
axes[1][0].set_title('r = T0 / 8')

tau = T_0 / 16
w = [n * w_0 for n in range(-width, width)]
y = [A * tau * w_0 * Sa((n * w_0 * tau) / 2) for n in range(-width, width)]
axes[1][1].stem(w, y)
axes[1][1].set_title('r = T0 / 16')
'''
tau = T_0 / 1000
w = [n * w_0 for n in range(-width, width)]
y = [A * tau * w_0 * Sa((n * w_0 * tau) / 2) for n in range(-width, width)]
plt.stem(w, y)
plt.title('r = T0 / 1000')
plt.show()