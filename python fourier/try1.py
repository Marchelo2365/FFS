import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
import scipy.io.wavfile

# Read the audio file
rate, data = scipy.io.wavfile.read('audio.wav')

# Perform Fourier transform
fourier = np.fft.fft(data)

# Create a frequency axis (only positive frequencies)
n = len(data)
freq = np.fft.fftfreq(n, d=1/rate)[:n//2]

# Plot the Fourier transform
plt.plot(freq, np.abs(fourier[:n//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# Set custom x-axis ticks
xticks = np.arange(0, rate/2, 500)  # Adjust the step size (500 Hz in this case)
plt.xticks(xticks)

# Format x-axis tick labels to display whole numbers
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_major_formatter().set_scientific(False)

# Format y-axis tick labels to display whole numbers
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# Display the plot
plt.show()
