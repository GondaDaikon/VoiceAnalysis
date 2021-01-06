import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

F1 = np.loadtxt('F1.txt')
F2 = np.loadtxt('F2.txt')

plt.figure()
plt.plot(F1, F2)
plt.scatter(F1, F2)
plt.axis([0, 1000, 0, 3000])
plt.savefig('01.png')
