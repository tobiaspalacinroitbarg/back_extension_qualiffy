# Librerías
import torch
import subprocess
import whisper
import wave
import contextlib
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import datetime
import warnings
import time
from langchain_ollama import OllamaLLM

# Suprimir los warnings de tipo FutureWarning y UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def convert_mono(audio_path:str):
        temp_path = f'{audio_path}_mono.wav'
        print(f"Convirtiendo {audio_path} a mono (si es necesario)...")
        # Convertir a mono con FFmpeg, sin importar el número de canales
        subprocess.call(['ffmpeg', '-i', audio_path, '-ac', '1', temp_path, '-y'])
        return temp_path

def transcribe_diarized(path:str,num_speakers:int, language:str, model_size:str):
    # Suprimir los warnings de tipo FutureWarning y UserWarning
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Iniciar el timer
    start_time = time.time()

    # Selección del nombre del modelo
    model_name = model_size
    if language == 'English' and model_size != 'large':
        model_name += '.en'

    # Asegurarse de que el archivo de audio es mono
    path = convert_mono(path)

    # Cargar el modelo de embeddings de voz
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Comprobar si el archivo es .wav, convertir si es necesario
    if path[-3:] != 'wav':
        subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
        path = 'audio.wav'

    # Cargar el modelo de Whisper
    model = whisper.load_model(model_size)

    # Transcribir el archivo de audio
    result = model.transcribe(path)
    segments = result["segments"]

    # Obtener la duración del audio
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Función para extraer el embedding de cada segmento
    audio = Audio()

    def segment_embedding(segment):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(path, clip)
        return embedding_model(waveform[None])

    # Calcular embeddings para cada segmento
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    embeddings = np.nan_to_num(embeddings)

    # Agrupamiento para asignar oradores
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    # Función para convertir tiempo en formato legible
    def time_format(secs):
        return str(datetime.timedelta(seconds=round(secs)))

    # Guardar el resultado en un archivo de texto
    with open("/media/transcript.txt", "w") as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + time_format(segment["start"]) + '\n')
            f.write(segment["text"][1:] + ' ')

    # Fin del timer y mostrar tiempo de ejecución
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"El tiempo total de transcripción y diarización fue de {execution_time:.2f} segundos.")
    return

def generar_consejo_con_ollama(transcription_path:str, datos_vendedor:dict):
    start_time = time.time()
    with open(transcription_path, 'r', encoding='utf-8') as file:
        transcription_text = file.read()
    model = OllamaLLM(model="llama3")   
    prompt = f"Eres un asistente y tienes que dar consejos de una sola oración (máximo 15 palabras por consejo, y si es más breve mejor) a partir de una transcripción de una llamada de ventas. Solo puedes hablar de dos maneras: La primera es respondiendo: 'Consejo IA: (El consejo que deberás dar)', si crees que puede haber un consejo para el vendedor en relación a cómo está hablando, si debe corregir su tono, claridad, etc.  La segunda manera en la que puedes hablar es diciendo 'Asesor IA: (problema:resolución)', si encuentras un problema que está mencionando el cliente y el vendedor lo puede resolver con sus servicios. Te paso los datos: \nTranscripción: {transcription_text}\n ,Datos del vendedor: {datos_vendedor}\n"
    result= model.invoke(input=prompt)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"El tiempo total de la generación del consejo fue de {execution_time:.2f} segundos.")
    print(result)
    return result