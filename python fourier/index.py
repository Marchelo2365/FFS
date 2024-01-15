import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Función para cargar y preprocesar archivos de audio
def load_and_preprocess_audio(file_path, sample_rate=22050, duration=5):
    audio, _ = librosa.load(file_path, sr=sample_rate, duration=duration)
    # Aplicar transformada de Fourier para obtener la representación espectral
    stft = librosa.stft(audio)
    # Obtener el módulo del espectrograma
    magnitude = np.abs(stft)
    # Convertir a decibeles
    log_magnitude = librosa.amplitude_to_db(magnitude)
    # Asegurarse de que el espectrograma tiene la forma adecuada
    log_magnitude = np.expand_dims(log_magnitude, axis=-1)
    return log_magnitude

# Función para construir el modelo CNN
def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Puedes ajustar según tus necesidades
    return model

# Ruta de tu archivo de audio
audio_file_path = 'prueba.wav'

# Cargar y preprocesar el audio
input_data = load_and_preprocess_audio(audio_file_path)

# Definir la forma de entrada del modelo
input_shape = input_data.shape

# Construir el modelo
model = build_cnn_model(input_shape)

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo (puedes ajustar el número de épocas y el tamaño de lote según tus necesidades)
model.fit(np.expand_dims(input_data, axis=0), np.array([1.0]), epochs=10, batch_size=1)

# Guardar el modelo entrenado (opcional)
model.save('voice_replication_model.h5')
