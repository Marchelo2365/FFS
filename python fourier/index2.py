import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pyttsx3

# Función para cargar y preprocesar archivos de audio (la misma función que antes)
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

# Función para construir el modelo CNN (la misma función que antes)
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

# Ruta de tu archivo de audio para hacer predicciones
new_audio_file_path = 'prueba.wav'

# Cargar y preprocesar el nuevo audio
new_input_data = load_and_preprocess_audio(new_audio_file_path)

# Cargar el modelo entrenado
loaded_model = models.load_model('voice_replication_model.h5')

# Hacer predicciones con el nuevo audio
predictions = loaded_model.predict(np.expand_dims(new_input_data, axis=0))

# Convertir las predicciones en texto (esto es un ejemplo, puedes ajustarlo según tus necesidades)
generated_text = "La voz replicada podría decir: {}".format(predictions)

# Utilizar pyttsx3 para sintetizar la voz y reproducirla
engine = pyttsx3.init()
engine.say(generated_text)
engine.runAndWait()
