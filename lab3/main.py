import numpy as np
import matplotlib.pyplot as plt

# 时长为1秒
t = 1
# 采样率为60hz
fs = 60
t_split = np.arange(0, t * fs)

# 1hz与25hz叠加的正弦信号
x_1hz = t_split * 1 * np.pi * 2 / fs
x_25hz = t_split * 25 * np.pi * 2 / fs
signal_sin_1hz = np.sin(x_1hz)
signal_sin_25hz = np.sin(x_25hz)

signal_sin = signal_sin_1hz + 0.25 * signal_sin_25hz


def filter_fir(input):
    omiga_c = 2 * np.pi * 16 / fs
    h_w = []
    for n in range(-8, 9):
        if n:
            h = np.sin(n * omiga_c) / (n * np.pi)
        else:
            h = omiga_c / np.pi
        h_w.append(h * (0.5 + 0.5 * np.cos(2 * np.pi * n / 16)))
    return np.convolve(input, h_w)[: 60]


def filter_zero_phase(input):
    input = filter_fir(input)
    input = input[:: -1]
    input = filter_fir(input)
    input = input[:: -1]
    return input


if __name__ == "__main__":
    delay_filtered_signal = filter_fir(signal_sin)
    zerophase_filtered_signal = filter_zero_phase(signal_sin)

    plt.plot(t_split, signal_sin, label='origin')
    plt.plot(t_split, delay_filtered_signal, label='fir')
    plt.plot(t_split, zerophase_filtered_signal, label='zero phase')

    plt.show()
