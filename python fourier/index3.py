import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Función para cargar y preprocesar archivos de audio
def load_and_preprocess_audio(file_path, sample_rate=22050, duration=5):
    audio, _ = librosa.load(file_path, sr=sample_rate, duration=duration)
    # Aplicar transformada de Fourier para obtener el espectro de frecuencia
    spectrum = np.abs(librosa.stft(audio))
    return spectrum

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
    model.add(layers.Dense(spectrum.shape[0], activation='linear'))  # Salida con la misma dimensión del espectro
    return model

# Ruta de tu archivo de audio para hacer predicciones
new_audio_file_path = 'prueba.wav'

# Cargar y preprocesar el nuevo audio
new_input_data = load_and_preprocess_audio(new_audio_file_path)

# Cargar el modelo entrenado
loaded_model = models.load_model('voice_replication_model.h5')

# Hacer predicciones con el nuevo audio
generated_spectrum = loaded_model.predict(np.expand_dims(new_input_data, axis=0))

# Configurar hop_length
hop_length = 512  # Puedes ajustar este valor según tus necesidades

# Convertir las características espectrales en audio
generated_audio = librosa.istft(np.transpose(generated_spectrum), hop_length=hop_length)

# Reproducir el audio generado
librosa.output.write_wav('voz_replicada.wav', generated_audio, sr=22050)
