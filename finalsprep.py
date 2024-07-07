# import numpy as np
# import matplotlib.pyplot as plt


# def bpsk():
#     t = np.arange(0, 4*np.pi, 0.1)
#     msg_f = 20
#     carrier_f = 30
#     msg = np.sign(np.sin(2 * np.pi * msg_f * t))
#     carrier = np.cos(2 * np.pi * carrier_f * t)

#     fig, axs = plt.subplots(2,1)

#     axs[0].plot(t, msg)
#     axs[1].plot(t, carrier)
#     plt.show()


# if __name__ == '__main__':
#     bpsk()


import numpy as np
import matplotlib.pyplot as plt


def bpsk():
    # Higher resolution for better visualization
    t = np.arange(0, 4*np.pi, 0.01)
    # Adjusted to fit within the time array => time period is pi multiple
    msg_f = 1/(np.pi)
    carrier_f = 5 * msg_f  # Carrier frequency is a multiple of the message frequency

    msg = np.sign(np.sin(2 * np.pi * msg_f * t))
    carrier = np.cos(2 * np.pi * carrier_f * t)
    modulated = msg * carrier

    fig, axs = plt.subplots(3, 1, figsize=(10, 6))

    axs[0].plot(t, msg)
    axs[0].set_title("Message Signal")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")

    axs[1].plot(t, carrier)
    axs[1].set_title("Carrier Signal")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Amplitude")

    axs[2].plot(t, modulated)
    axs[2].set_title("Modulated Signal (BPSK)")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def qpsk():
    def cosinewave_phase(phase, num_samples=100):
        # Convert phase to radians
        phase = phase * (np.pi / 180)
        t = np.linspace(0, np.pi, num_samples)
        cosine = np.cos(t + phase)
        return cosine

    # Generate random binary message
    message = '1110010001000111'
    message = [str(int(x)) for x in message]
    message = "".join(message)
    message = [message[2*i: 2*(i+1)] for i in range(0, len(message)//2)]

    # Phase dictionary
    dict_phases = {'00': 0, '01': 90, '10': 180, '11': 270}

    # Modulate the message
    modulated = []
    for symbol in message:
        modulated.extend(cosinewave_phase(dict_phases[symbol]))

    # Create time array for plotting
    t = np.arange(0, len(modulated), 1)

    fig, axs = plt.subplots(2, 1)
    # Plot the binary message signal

    # Plot the binary message signal
    bits = [int(bit) for bit in "".join(message)]
    samples_per_bit = 50  # Adjust this for desired width
    x = np.arange(len(bits) * samples_per_bit)
    y = np.repeat(bits, samples_per_bit)

    # Plot the signal
    axs[0].plot(x, y, drawstyle='steps-pre', linewidth=2)

    # Add vertical lines to separate 2-bit groups
    for i in range(0, len(bits), 2):
        axs[0].axvline(x=i*samples_per_bit, color='r',
                       linestyle='--', alpha=0.5)

    # Customize the plot
    axs[0].set_title('Binary Message Signal')
    axs[0].set_xlabel('Sample Number')
    axs[0].set_ylabel('Bit Value')
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels(['0', '1'])
    axs[0].grid(True, which='major', linestyle='-', alpha=0.6)
    axs[0].grid(True, which='minor', linestyle=':', alpha=0.3)
    axs[0].set_ylim(-0.1, 1.1)

    # Annotate 2-bit groups
    for i in range(0, len(bits), 2):
        axs[0].text(i*samples_per_bit + samples_per_bit, 1.15,
                    f'{bits[i]}{bits[i+1]}', ha='center', va='center')

    # Plot the modulated signal
    axs[1].plot(t, modulated)
    axs[1].set_title('QPSK Modulated Signal')
    axs[1].set_xlabel('Sample Number')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)
    fig.tight_layout()
    plt.show()


def bersim():
    N = 5000
    EbN0_list = np.arange(0, 50)
    BER = []
    for EbN0 in EbN0_list:
        EbN0 = 10**(EbN0/10)
        x = 2 * (np.random.randn(N) >= 0.5) - 1
        noise = 1 / np.sqrt(2*EbN0)
        chanel = x + np.random.randn(N)*noise
        received = 2 * (chanel >= 0.5) - 1
        error = (x != received).sum()
        BER.append(error/N)

    plt.plot(EbN0_list, BER, "-", EbN0_list, BER, "go")
    plt.axis([0, 14, 1e-7, 0.1])
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid()
    plt.xlabel("EbN0 in dB")
    plt.ylabel("BER")
    plt.title("BER in BPSK")
    plt.show()
    plt.xscale('linear')
    plt.yscale('log')
    plt.axis([0, 14, 1e-7, 0.1])

    plt.show()


if __name__ == '__main__':
    # bpsk()
    # bersim()
    qpsk()
