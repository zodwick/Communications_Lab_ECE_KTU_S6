import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_symbols = 100  # Increased for better eye diagram
sps = 8
beta = 0.35
output_length = num_symbols * sps  # Changed to accommodate all symbols
SNR_dB = 10

# Generate random bits
bits = np.random.randint(0, 2, num_symbols)

# Create the pulse
x = np.array([np.concatenate(([bit*2-1], np.zeros(sps-1)))
             for bit in bits]).flatten()

# Create the raised-cosine filter
Ts = sps
t = np.arange(-50, 51)

h = (np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)) * (1/Ts)

# Convolve the signal with the filter
x_shaped = np.convolve(x, h)

# Trim the convolution result to match the desired output length
start_index = (len(x_shaped) - output_length) // 2
end_index = start_index + output_length
x_shaped_trimmed = x_shaped[start_index:end_index]
# x_shaped_trimmed = np.convolve(x, h, mode='same')
# Add Gaussian noise
power_signal = np.mean(np.abs(x_shaped_trimmed) ** 2)
power_noise = power_signal / (10 ** (SNR_dB / 10))
# print(len(x_shaped_trimmed))
# noise = np.random.normal(0, np.sqrt(power_noise), len(x_shaped_trimmed))
noise = np.random.randn(len(x_shaped_trimmed)) * np.sqrt(power_noise)
x_noisy = x_shaped_trimmed + noise

# Sample the signal
sampled_indices = np.arange(0, len(x_noisy), sps)  # Sample every sps-th sample
# print(sampled_indices)
# print(len(x_noisy))

x_sampled = x_noisy[sampled_indices]

threshold = 0
# print(len(x_shaped_trimmed))
# print(len(x_sampled))

# Make decisions based on threshold
decisions = np.where(x_sampled > threshold, 1, 0)

# Create eye diagram


def create_eye_diagram(signal, sps):
    eye_length = 2 * sps
    n_samples = len(signal)
    n_traces = n_samples // eye_length
    # print(n_samples, "n_samples")
    print(n_traces, "n_traces")
    print(eye_length, "eye_length")
    # eye_diagram = np.zeros((eye_length, n_traces))
    eye_diagram = np.zeros((eye_length, n_traces))
    print(eye_diagram)
    # print(eye_diagram.shape)

    for i in range(n_traces):
        start = i * eye_length
        end = start + eye_length
        # print(start, end)
        eye_diagram[:, i] = signal[start:end]

    return eye_diagram


eye_diagram = create_eye_diagram(x_shaped_trimmed, sps)
eye_diagram_noisy = create_eye_diagram(x_noisy, sps)

# Plotting
plt.figure(figsize=(25, 22))

# Plot original pulse
plt.subplot(4, 2, 1)
plt.plot(x[:8*sps], '.-')
plt.title('Original Pulse (first 8 symbols)')
plt.grid(True)

# Plot raised-cosine filter
plt.subplot(4, 2, 2)
plt.plot(t, h, '.')
plt.title('Raised-Cosine Filter')
plt.grid(True)

# Plot convolved signal
plt.subplot(4, 2, 3)
plt.plot(x_shaped[:8*sps], '.-')
plt.title('Convolved Signal (first 8 symbols)')
plt.grid(True)

# Plot signal with added noise
plt.subplot(4, 2, 4)
plt.plot(x_noisy[:8*sps], '.-')
plt.title('Signal with AWGN (first 8 symbols)')
plt.grid(True)

# Plot sampled signal
plt.subplot(4, 2, 5)
plt.plot(sampled_indices[:8], x_sampled[:8], '.-')
plt.title('Sampled Signal (first 8 symbols)')
plt.grid(True)

# Plot decisions
plt.subplot(4, 2, 6)
plt.step(sampled_indices[:8], decisions[:8],
         'g', where='mid', label='Decisions')
plt.title('Decisions (first 8 symbols)')
plt.xlabel('Time')
plt.ylabel('Amplitude/Decision')
plt.grid(True)

# Plot eye diagram
plt.subplot(4, 2, 7)
plt.plot(eye_diagram)
plt.title('Eye Diagram')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot noisy eye diagram
plt.subplot(4, 2, 8)
plt.plot(eye_diagram_noisy)
plt.title('Noisy Eye Diagram')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)


plt.tight_layout()
plt.show()
