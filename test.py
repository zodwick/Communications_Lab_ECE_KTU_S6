import matplotlib.pyplot as plt
import numpy as np


# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
# fig.subplots_adjust(hspace=0.4)

t = np.arange(0, 2, 0.001)
print(len(t))
symbols = (np.random.randint(0, 2, 20)+1)
print(symbols)
train = np.repeat(symbols, 100)
print(len(symbols))
print(train)
print(len(train))

# plt.plot(t, symbols)

f = 10
x = np.sin(2*np.pi*f*t)*train
plt.plot(t, train)
plt.plot(t, x)


plt.show()
