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
    N = 50000
    EbNo_list = np.arange(0, 20, 1)
    BER = []

    for EbNoDB in EbNo_list:
        Ebn0 = 10 ** (EbNoDB/10)
        msg = 2 * (np.random.rand(N) >= 0.5) - 1

        noise_std_dev = 1 / np.sqrt(2 * Ebn0)
        channel = msg + np.random.randn(len(msg))*noise_std_dev

       # print(min(channel), max(channel))
        received = 2 * (channel >= 0) - 1
        error = (msg != received).sum()
        BER.append(error/N)

    print(BER)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(EbNo_list, BER, '-', EbNo_list, BER, 'go-')
    axs[0].set_xlabel('EbN0')
    axs[0].set_ylabel('BER')
    axs[0].set_title(' EbN0 v/s BER for BPSK')
    # axs[0].tight_layout()

    axs[1].semilogy(EbNo_list, BER, '-', EbNo_list, BER, 'go-')
    axs[1].set_xlabel('EbN0')
    axs[1].set_ylabel('BER')
    axs[1].set_title(' EbN0 v/s BER for BPSK')
    plt.show()


def qpskber():
    N = 50000
    EbNo_list = np.arange(0, 10, 1)
    BER = []
    c = 0

    for EbNoDB in EbNo_list:
        Ebn0 = 10 ** (EbNoDB/10)
        x = np.random.rand(N) >= 0
        x_str = [str(int(i)) for i in x]
        x_str = "".join(x_str)
        message = [x_str[2*i: 2*(i+1)] for i in range(int(len(x)/2))]
        noise = 1/np.sqrt(2 * Ebn0)
        channel = x + np.random.randn(N) * noise
        received_x = channel >= 0.5
        xReceived_str = [str(int(i)) for i in received_x]
        xReceived_str = "".join(xReceived_str)
        messageReceived = [xReceived_str[2*i: 2 *
                                         (i+1)] for i in range(int(len(x)/2))]

        message2 = np.array(message)
        messageReceived = np.array(messageReceived)
        errors = (message2 != messageReceived).sum()

        # if c == 0:
        #     c += 1
        #     print("x: ", x)
        #     print("message: ", len(message))
        #     print("channel :", channel)
        #     print(max(channel), min(channel))
        #     print(received_x)
        #     print("message2: ", message2)

        BER.append(errors/N)

    print(BER)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(EbNo_list, BER, '-', EbNo_list, BER, 'go-')
    axs[0].set_xlabel('EbN0')
    axs[0].set_ylabel('BER')
    axs[0].set_title(' EbN0 v/s BER for BPSK')
    # axs[0].tight_layout()

    axs[1].semilogy(EbNo_list, BER, '-', EbNo_list, BER, 'go-')
    axs[1].set_xlabel('EbN0')
    axs[1].set_ylabel('BER')
    axs[1].set_title(' EbN0 v/s BER for BPSK')
    plt.show()


def pcm():
    # analog

    fig, axs = plt.subplots(4, 1)
    t = np.arange(0, 0.1, 0.0001)
    msg_f = 100
    dc = 2
    sig = np.sin(2*np.pi * t * msg_f)+dc
    axs[0].plot(t, sig)
    axs[0].set_title("signal")

    # Sampling
    fs = 8 * msg_f
    ts = np.arange(0, 0.1, 1/fs)
    sampled_s = dc + np.sin(2*np.pi * msg_f * ts)
    axs[1].plot(ts, sampled_s)
    axs[1].set_title("signal")

    # Quantising

    L = 16
    sig_min = np.min(sig)
    sig_max = np.max(sig)
    q_levels = np.linspace(sig_min, sig_max, L)  # indexes with length of msg
    q_sig = np.digitize(sampled_s, q_levels, right='true')
    q_sig = q_levels[q_sig]  # each index is replaced with the value

    axs[2].plot(ts, q_sig, "b", ts, q_sig, "m*")
    axs[2].set_title("Quantized Signal")

    def power(s):
        return np.mean(np.square(s))

    q_noise = q_sig - sampled_s
    axs[3].plot(ts, q_noise)
    axs[3].set_title("Quantization Noise")
    # plt.hist(q_noise)
    plt.show()

    p_signal = power(sig)
    p_noise = power(q_noise)
    snr = p_signal / p_noise
    snr_db = 20 * np.log10(snr)
    print("Signal-to-Noise ratio in dB:", snr_db)

    snr_db_list = []

    for R in range(1, 11):
        snr_db = 6.02 * R + 1.76
        snr_db_list.append(snr_db)

    plt.plot(range(1, 11), snr_db_list, "r*-")
    plt.xlabel("Number of Bits per Symbol")
    plt.ylabel("SNR in dB")
    plt.title("SNR vs Number of Bits per Symbol")
    plt.show()


def matched_filter():
    print("Matched Filter")


if __name__ == '__main__':
    # bpsk()
    # bersim()
    # qpsk()
    # qpskber()
    # pcm()
    matched_filter()
