import numpy as np
import matplotlib.pyplot as plt

# plotting msg signal - sin
fm = 10
dc_offset = 2
fs = 30 * fm
t = np.arange(0, 4/fm, 0.0001)
message = np.cos(2*np.pi*fm*t)
plt.plot(t, message)
plt.show()


# sampling the msg signal
ts = np.arange(0, 4/fm, 1/fs)
message_sampled = np.cos(2*np.pi*fm*ts)
plt.plot(ts, message_sampled, "bo-")
plt.show()


# quantisation
L = 8  # 3 bit pcm
x_min = min(message)
x_max = max(message)
quantisationlevels = np.linspace(x_min, x_max, L)


qinp = np.linspace(x_min, x_max, 1000)
qout = []
xquantised = []


# quantising input signal by dividing them into 1000 levels
for i in qinp:
    for j in quantisationlevels:
        if i <= j:
            qout.append(j)
            break
quot2 = np.digitize(qinp, quantisationlevels, right='true')

fig, axs = plt.subplots(2, 1)

axs[0].plot(qinp, qout)
axs[0].set_title("qout")

axs[1].plot(qinp, quot2)
axs[1].set_title("digitse np")
plt.show()


# # quantising sampled signal
for i in message_sampled:
    for j in quantisationlevels:
        if i <= j:
            xquantised.append(j)
            break

# xquantised = np.digitize(message_sampled, quantisationlevels, right='true')
plt.plot(ts, xquantised, "bo-")
plt.title("xquantisse")

plt.show()


# SNR
def power(lst):
    P = 0
    for i in lst:
        P = P + i**2
    return P/len(lst)


quantizationNoise = xquantised - message_sampled
snr = power(message_sampled) / power(quantizationNoise)
snrdB = 20 * np.log10(snr)
print(snrdB)
