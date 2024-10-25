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
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
import re

# Suprimir los warnings de tipo FutureWarning y UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Descargar las stopwords si no las tienes
nltk.download('punkt')
nltk.download('punkt_tab')
# Descargar stopwords si no las tienes ya
nltk.download('stopwords')

def extract_ia_sections(text):
    # Definir patrones para Asesor IA, Consejo IA y Resumen IA
    asesor_pattern = re.compile(r"Asesor IA:\s*(.*?)\n", re.DOTALL)
    consejo_pattern = re.compile(r"Consejo IA:\s*(.*?)\n", re.DOTALL)
    resumen_pattern = re.compile(r"Resumen IA:\s*(.*?)$", re.DOTALL)

    # Buscar coincidencias en el texto
    asesor_match = asesor_pattern.search(text)
    consejo_match = consejo_pattern.search(text)
    resumen_match = resumen_pattern.search(text)

    # Extraer el contenido si hay coincidencias, sino devolver None
    asesor_ia = asesor_match.group(1).strip() if asesor_match else None
    consejo_ia = consejo_match.group(1).strip() if consejo_match else None
    resumen_ia = resumen_match.group(1).strip() if resumen_match else None

    return asesor_ia, consejo_ia, resumen_ia

def read_file_with_fallback_encoding(file_path):
    """
    Intenta leer un archivo con diferentes codificaciones.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    
    raise UnicodeDecodeError(f"No se pudo leer el archivo con ninguna de las codificaciones: {encodings}")

def generar_consejo_con_ollama(transcription_path: str, datos_vendedor: dict):
    start_time = time.time()
    try:
        transcription_text = read_file_with_fallback_encoding(transcription_path)
        model = OllamaLLM(model="llama3")   
        prompt = f"""Eres un asistente y tienes que dar consejos de una sola oración (máximo 15 palabras por consejo, 
        y si es más breve mejor) a partir de una transcripción de una llamada de ventas. 
        Solo puedes hablar de tres maneras: 
        La primera es respondiendo: 'Consejo IA: (El consejo que deberás dar, no uses vocativos, solo da el consejo en tiempo presente o imperativo)', 
        si crees que puede haber un consejo para el vendedor en relación a cómo está hablando, si debe corregir su tono, claridad, hacer alguna pregunta en particular, etc.  
        La segunda manera en la que puedes hablar es diciendo 'Asesor IA: (problema:resolución)', 
        si encuentras un problema que está mencionando el cliente y el vendedor lo puede resolver con sus servicios. 
        Por último, debes obligatoriamente generar un resumen en menos de 50 palabras de la reunión y devolverlo en la siguiente estructura 
        'Resumen IA: (resumen de la reunión)'
        [Te paso los datos: 
        Transcripción: {transcription_text}
        Datos del vendedor: {datos_vendedor}]"""
        
        result = model.invoke(input=prompt)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"El tiempo total de la generación del consejo fue de {execution_time:.2f} segundos.")
        print(result)
        return result
    except Exception as e:
        print(f"Error en generar_consejo_con_ollama: {str(e)}")
        raise

def calculate_word_percentage(file_path: str, speaker: str = "SPEAKER 1"):
    try:
        # Leer el archivo con el manejo de codificación mejorado
        transcript = read_file_with_fallback_encoding(file_path)
        
        # Regex modificado para capturar el formato específico
        speaker_pattern = re.compile(rf"{speaker} \d+:\d+:\d+\n(.*?)(?=SPEAKER|\Z)", re.DOTALL)
        
        # Obtener todos los diálogos del speaker elegido
        speaker_dialogues = speaker_pattern.findall(transcript)
        
        if not speaker_dialogues:
            print(f"No se encontraron diálogos para {speaker}")
            return 0.0
            
        # Concatenar todo el diálogo de ese speaker en un solo texto
        speaker_text = " ".join(speaker_dialogues)
        
        # Tokenizar el texto del speaker
        speaker_words = nltk.word_tokenize(speaker_text)
        
        # Tokenizar todo el transcript
        all_words = nltk.word_tokenize(transcript)
        
        # Filtrar las palabras no vacías y sin puntuación
        speaker_words = [word for word in speaker_words if any(c.isalnum() for c in word)]
        all_words = [word for word in all_words if any(c.isalnum() for c in word)]
        
        # Calcular el porcentaje
        if len(all_words) > 0:
            percentage_spoken = (len(speaker_words) / len(all_words)) * 100
        else:
            percentage_spoken = 0
            
        print(f"Speaker words: {len(speaker_words)}, All words: {len(all_words)}, Percentage: {percentage_spoken}")
        return round(percentage_spoken, 2)
    except Exception as e:
        print(f"Error en calculate_word_percentage: {str(e)}")
        return 0.0

def top_words_from_transcript(file_path: str, top_n: int = 5, language: str = 'spanish'):
    # Leer el archivo
    transcript = read_file_with_fallback_encoding(file_path)
    
    # Asegurarse de que NLTK tiene las stopwords necesarias
    try:
        stop_words = set(stopwords.words(language))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words(language))
    
    # Tokenizar el texto
    try:
        words = nltk.word_tokenize(transcript.lower())
    except LookupError:
        nltk.download('punkt')
        words = nltk.word_tokenize(transcript.lower())
    
    # Filtrar stopwords y puntuación
    filtered_words = [
        word for word in words 
        if word not in stop_words 
        and word not in string.punctuation
        and any(c.isalnum() for c in word)
    ]
    
    # Contar frecuencia
    word_counts = Counter(filtered_words)
    
    # Obtener top N palabras
    most_common_words = word_counts.most_common(top_n)
    
    return most_common_words

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

    # Modificar la parte donde se guarda el archivo
    with open("media/transcript.txt", "w", encoding="utf-8", errors='ignore') as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + time_format(segment["start"]) + '\n')
            f.write(segment["text"][1:] + ' ')

    # Fin del timer y mostrar tiempo de ejecución
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"El tiempo total de transcripción y diarización fue de {execution_time:.2f} segundos.")
    return

