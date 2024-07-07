import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

# Generate the time vector for the continuous signal
t = np.arange(0, 0.1, 0.00001)

# Generate the message signal (sinusoidal)
message = np.sin(2 * np.pi * 100 * t)

# Plot the continuous message signal
plt.figure(figsize=(10, 4))
plt.plot(t, message)
plt.title("Continuous Message Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Generate the time vector for the sampled signal
ts = np.arange(0, 0.1, 1 / (2 * 1000))

# Sample the message signal
sampled_message = np.sin(2 * np.pi * 100 * ts)

# Plot the sampled message signal
plt.figure(figsize=(10, 4))
plt.plot(ts, sampled_message, "g", label="Sampled Signal")
plt.plot(ts, sampled_message, "r*", label="Samples")
plt.title("Sampled Message Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# Get the quantization level from the user
L = 16

print("sampled_message:", sampled_message)
# Generate quantization levels
tq = np.linspace(round(min(sampled_message)), round(max(sampled_message)), L)
print("Quantization levels:", tq)

# Quantize the sampled signal
q_sig = [min(tq, key=lambda x: abs(x - s)) for s in sampled_message]
print("Quantized signal:", q_sig)

# Plot the quantized signal
plt.figure(figsize=(10, 4))
plt.plot(ts, q_sig, "b", label="Quantized Signal")
plt.plot(ts, q_sig, "bo", label="Quantized Samples")
plt.title("Quantized Message Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# Create a dictionary to map quantized levels to binary codes
bit = int(np.ceil(np.log2(L)))
binary = {level: format(i, f'0{bit}b') for i, level in enumerate(tq)}
print("Binary encoding:", binary)
# Encode the quantized message
encoded_message = [binary[q] for q in q_sig]
print("Encoded message:", encoded_message)

# Compute the quantization noise
noise = np.array(q_sig) - np.array(sampled_message)

# Plot the quantization noise
plt.figure(figsize=(10, 4))
plt.plot(ts, noise, "g", label="Quantization Noise")
plt.title("Quantization Noise")
plt.xlabel("Time (s)")
plt.ylabel("Noise Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# Define a function to compute power of a signal


def power(signal):
    return np.mean(np.square(signal))


# Compute SNR
snr = power(sampled_message) / power(noise)
snr_db = 20 * np.log10(snr)
print("SNR (dB):", snr_db)


# Create a DataFrame
data = {
    "Sampled Message": sampled_message,
    "Quantized Signal": q_sig,
    "Encoded Message": encoded_message
}

df = pd.DataFrame(data)
print("Binary encoding:", binary)

# Display the DataFrame
print(df)
