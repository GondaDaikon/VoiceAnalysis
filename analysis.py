import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def Hanning_window(N):
    w = np.zeros(N)
    if N % 2 == 0:
        for n in range(N):
            w[n] = 0.5 - 0.5 * np.cos(2 * np.pi * n / N)
    else:
        for n in range(N):
            w[n] = 0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / N)
    return w

def Levinson_Durbin(s, lpc_order):
    length_of_s = len(s)
    a = np.zeros(lpc_order + 1)
    r = np.zeros(lpc_order + 1)
    lpc = np.zeros(lpc_order + 1)
    gamma = np.zeros(lpc_order + 1)
    parcor = np.zeros(lpc_order + 1)
    epsilon = np.zeros(lpc_order + 1)

    for m in range(lpc_order + 1):
        for n in range(length_of_s - m):
            r[m] += s[n] * s[n + m]

    epsilon[0] = r[0]
    gamma[1] = -r[1] / epsilon[0]
    lpc[0] = 1
    lpc[1] = gamma[1]
    epsilon[1] = epsilon[0] * (1 - gamma[1] * gamma[1])

    for k in range(2, lpc_order + 1):
        for m in range(k):
            a[m] = lpc[m]

        a[k] = 0
        num = 0
        for m in range(k):
            num += a[m] * r[k - m]

        gamma[k] = -num / epsilon[k - 1]
        for m in range(k + 1):
            lpc[m] = a[m] + gamma[k] * a[k - m]

        epsilon[k] = epsilon[k - 1] * (1 - gamma[k] * gamma[k])

    for m in range(lpc_order + 1):
        parcor[m] = -gamma[m]

    return lpc, parcor

fs, s0 = wavfile.read('sound/a.wav')
s0 = s0.astype(np.float)
length_of_s0 = len(s0)
for n in range(length_of_s0):
    s0[n] = s0[n] + 32768 + (np.random.rand() - 0.5) / 2
    s0[n] = (s0[n] - 32768) / 32768

# pre emphasis
d = np.zeros(length_of_s0)
for n in range(1, length_of_s0):
    d[n] = s0[n] - 0.98 * s0[n - 1]

lpc_order = 10
window_size = 256
shift_size = 160
N = 1024

number_of_frame = int((length_of_s0 - (window_size - shift_size)) / shift_size)

F1 = np.zeros(number_of_frame)
F2 = np.zeros(number_of_frame)

s = np.zeros(window_size)
w = Hanning_window(window_size)
x = np.zeros(N)
A = np.zeros(int(N / 2) + 1)

for frame in range(number_of_frame):
    offset = shift_size * frame
    for n in range(window_size):
        s[n] = d[offset + n] * w[n]

    lpc, parcor = Levinson_Durbin(s, lpc_order)

    for n in range(lpc_order + 1):
        x[n] = lpc[n]

    X = np.fft.fft(x, N)
    X_abs = np.abs(X)

    for k in range(int(N / 2 + 1)):
        A[k] = -20 * np.log10(X_abs[k])

    for k in range(1, int(N / 2) + 1):
        if A[k - 1] < A[k] and A[k] > A[k + 1]:
            F1[frame] = k
            break

    for k in range(int(F1[frame]) + 1, int(N / 2) + 1):
        if A[k - 1] < A[k] and A[k] > A[k + 1]:
            F2[frame] = k
            break

    F1[frame] = int(F1[frame] * fs / N)
    F2[frame] = int(F2[frame] * fs / N)

np.savetxt('F1.txt', F1)
np.savetxt('F2.txt', F2)
