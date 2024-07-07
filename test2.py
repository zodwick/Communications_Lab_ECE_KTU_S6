# PCM
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 1, 0.001)
messgae = np.sin(2*np.pi*t*20)

# plt.plot(t, messgae)
# plt.show()


fs = 400
t = np.arange(0, 1, 1/fs)
samples_msg = np.sin(2*np.pi*t*20)
# plt.plot(t, samples_msg, 'b', t, samples_msg, 'g*')
# plt.show()

print(samples_msg)

L = 16
pq = np.linspace(min(samples_msg), max(samples_msg), L)
print(pq)
print(len(pq))

quantised_signal = [min(pq, key=lambda x: abs(x-s))for s in samples_msg]
print(quantised_signal)


no_bits = int(np.ceil(np.log2(L)))
print(no_bits)
binanry = {level: format(i, f'0{no_bits}b')
           for i, level in enumerate(pq)}

encoded = [binanry[q] for q in quantised_signal]
noise = np.array(quantised_signal) - np.array(samples_msg)
print(noise)

print(binanry)
print(encoded)
