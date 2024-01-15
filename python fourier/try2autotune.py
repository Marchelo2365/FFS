import numpy as np
import tkinter as tk
import pyaudio
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import wave
from datetime import datetime

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK_SIZE = 1024

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)
# Global variables
audio_signal = np.zeros(CHUNK_SIZE, dtype=np.int16)  # <-- Error here
is_recording = False
frames = []
fundamental_frequency = 0.0  # Initialize to a default value

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)  # <-- This should be CHUNK_SIZE

# YIN Pitch Detection Algorithm
def compute_yin(signal):
    tau_max = len(signal) // 2
    yin_values = np.zeros(tau_max)

    for tau in range(1, tau_max):
        diff = signal[:-tau] - signal[tau:]
        squared_diff = np.square(diff)
        yin_values[tau] = np.sum(squared_diff)

    return yin_values

def estimate_fundamental_frequency(signal, rate):
    yin_values = compute_yin(signal)
    tau_candidates, _ = find_peaks(-yin_values)

    if len(tau_candidates) > 0:
        fundamental_frequency = rate / tau_candidates[0]
        return fundamental_frequency
    else:
        return 0.0

# GUI Setup
root = tk.Tk()
root.title("Real-Time Autotune")

label = tk.Label(root, text="Fundamental Frequency:")
label.pack()

fundamental_frequency_label = tk.Label(root, text="")
fundamental_frequency_label.pack()

# Real-time Changing Frequencies Graph
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, CHUNK_SIZE // 2)
ax.set_ylim(0, 2000)  # Adjust the y-axis limit based on your application
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Global variables
audio_signal = np.zeros(CHUNK_SIZE, dtype=np.int16)
is_recording = False
frames = []

# Function to update the GUI
def update_gui(frequency):
    fundamental_frequency_label.config(text=f"{frequency:.0f} Hz")
    root.update_idletasks()

# Function to update the plot
def update_plot(frame):
    global audio_signal
    data = stream.read(CHUNK_SIZE)
    audio_signal = np.frombuffer(data, dtype=np.int16)
    spectrum = np.abs(np.fft.fft(audio_signal))
    magnitudes = spectrum[:CHUNK_SIZE // 2]

    # Update the plot
    line.set_data(np.arange(0, CHUNK_SIZE // 2), magnitudes)
    return line,

ani = FuncAnimation(fig, update_plot, blit=True)

# Function to analyze the audio and update the GUI
def analyze_audio():
    global audio_signal, fundamental_frequency

    try:
        while True:
            # Read data from the microphone
            data = stream.read(CHUNK_SIZE)
            audio_signal = np.frombuffer(data, dtype=np.int16)

            # Estimate the fundamental frequency using YIN algorithm
            fundamental_frequency = estimate_fundamental_frequency(audio_signal, RATE)

            # Update the GUI
            update_gui(fundamental_frequency)

            # Update the plot
            spectrum = np.abs(np.fft.fft(audio_signal))
            magnitudes = spectrum[:CHUNK_SIZE // 2]
            line.set_data(np.arange(0, CHUNK_SIZE // 2), magnitudes)
            canvas.draw()

            # Record audio if the recording flag is set
            if is_recording:
                frames.append(data)

    except KeyboardInterrupt:
        print("Stopped by the user.")

    finally:
        # Close the stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

# Function to start and stop recording
def toggle_recording():
    global is_recording
    if not is_recording:
        is_recording = True
        record_button.config(text="Stop Recording")
    else:
        is_recording = False
        record_button.config(text="Start Recording")
        save_recording(frames)

# Function to save the recording
# Function to save the recording
def save_recording(frames):
    global fundamental_frequency
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")

    # Save audio recording in WAV format
    audio_filename = f"recording_{timestamp}.wav"
    wave_file = wave.open(audio_filename, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(p.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    # Save information in a text file
    text_filename = f"info_{timestamp}.txt"
    with open(text_filename, 'w') as f:
        f.write(f"Recording Timestamp: {timestamp}\n")
        f.write(f"Fundamental Frequency: {fundamental_frequency:.0f} Hz\n")
        # Add any other information you want to save

# Function to start and stop recording
def toggle_recording():
    global is_recording, fundamental_frequency
    if not is_recording:
        is_recording = True
        record_button.config(text="Stop Recording")
    else:
        is_recording = False
        record_button.config(text="Start Recording")
        save_recording(frames)

# Rest of the code remains the same.


# GUI elements
record_button = tk.Button(root, text="Start Recording", command=toggle_recording)
record_button.pack()

# Run the GUI
root.mainloop()
