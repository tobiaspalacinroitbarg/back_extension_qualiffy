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

def extract_relevant_parts(transcription_text: str, max_words: int = 150) -> str:
    """
    Extrae las partes más relevantes de la transcripción para reducir el tamaño del input.
    """
    # Eliminar timestamps y marcadores de speaker
    cleaned_text = re.sub(r'SPEAKER \d \d{2}:\d{2}:\d{2}\n', '', transcription_text)
    
    # Dividir en oraciones
    sentences = cleaned_text.split('.')
    
    # Seleccionar las oraciones más relevantes (primeras, últimas y algunas del medio)
    if len(sentences) > 10:
        selected = sentences[:3] + sentences[len(sentences)//2-1:len(sentences)//2+2] + sentences[-3:]
        return '. '.join(selected)
    return cleaned_text

def format_vendor_data(datos_vendedor: dict) -> str:
    """
    Formatea los datos del vendedor de manera concisa.
    """
    return ", ".join([f"{k}: {str(v)}" for k, v in datos_vendedor.items() if v])

def generate_ollama(transcription_path: str, datos_vendedor: dict, max_retries: int = 2, timeout: int = 30) -> str:
    """
    Versión optimizada de la función de generación de consejos.
    """
    start_time = time.time()
    
    try:
        # Leer y procesar la transcripción
        with open(transcription_path, 'r', encoding='utf-8') as file:
            full_transcription = file.read()
        
        # Extraer partes relevantes para reducir el tamaño del input
        relevant_text = extract_relevant_parts(full_transcription)
        
        # Formatear datos del vendedor de manera concisa
        vendor_info = format_vendor_data(datos_vendedor)
        
        # Crear una instancia del modelo con timeout
        model = OllamaLLM(
            model="llama3",
            temperature=0.3,  # Reducir temperatura para respuestas más concisas
            timeout=timeout
        )
        
        # Prompt optimizado y más conciso
        prompt = f"""Analiza esta llamada de ventas y proporciona si identificas
**Problema:**(texto máx 10 palabras)\n**Solución:**(texto de máx 10 palabras).

Contexto:
Transcripción relevante: {relevant_text}
Datos vendedor: {vendor_info}"""

        result = model.invoke(input=prompt)
        print(result)
        match = re.search(r'\*\*Problema:\*\* (.*?)\n\n\*\*Solución:\*\* (.*)', result)
        if match:
            problema = match.group(1)
            solucion = match.group(2)
            execution_time = time.time() - start_time
            print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
            return problema, solucion
        else:
            match = re.search(r'\*\*Problema:\*\* (.*?)\n\*\*Solución:\*\* (.*)', result)
            if match:
                problema = match.group(1)
                solucion = match.group(2)
                execution_time = time.time() - start_time
                print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
                return problema, solucion
            else:
                return "",""
        

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

def dict_consejos():
    """
    Crea un diccionario que mapea múltiples expresiones o patrones de habla con consejos de mejora.
    Las expresiones están agrupadas por el consejo que deberían trigger.
    """
    return {
        # Velocidad y fluidez del habla
        "Habla más despacio y organiza mejor tus ideas": [
            "um", "eh", "este", "emmm", "ahh", "uhh",
            "como te digo", "como te explico", "a ver déjame ver",
            "deja pienso", "mmm", "ehh", "am"
        ],
        
        # Claridad y precisión
        "Sé más específico y directo en tus explicaciones": [
            "como que", "más o menos", "digamos", "tal vez",
            "quizás", "a lo mejor", "puede ser que",
            "algo así", "una cosa así", "por ahí", "ponle que",
            "digamos que", "se podría decir"
        ],
        
        # Vocabulario profesional
        "Utiliza un vocabulario más profesional y formal": [
            "wey", "chido", "padre", "no mames", "nel",
            "órale", "va", "sale", "chale", "nel pastel",
            "equis", "neta", "simón", "fierro", "chamba",
            "chambear", "mande", "tons", "pos"
        ],
        
        # Muletillas a evitar
        "Evita el uso excesivo de muletillas": [
            "o sea", "básicamente", "literalmente", "prácticamente",
            "realmente", "sinceramente", "honestamente",
            "la verdad", "la neta", "fíjate que", "¿me explico?",
            "¿sí me entiendes?", "¿verdad?", "¿ok?"
        ],
        
        # Cortesía y engagement
        "Muestra más interés y elabora mejor tus respuestas": [
            "ajá", "pues", "ya", "sí", "no", "ok",
            "está bien", "como sea", "da igual",
            "lo que sea", "ni modo", "qué más da"
        ],
        
        # Precisión técnica
        "Utiliza términos más específicos y técnicos": [
            "cosa", "eso", "esto", "aquello", "así",
            "hacer algo", "hacer eso", "hacer esto",
            "el asunto", "la cosa esa", "el tema",
            "la situación", "el problema", "la cuestión"
        ],
        
        # Control de volumen y tono
        "Mantén un tono de voz profesional y moderado": [
            "!", "¡", "???", "¿¿¿",
            "MAYÚSCULAS", "¡¡", "!!", "???"
        ],
        
        # Interrupciones
        "Evita interrumpir y permite que otros terminen de hablar": [
            "pero pero", "espera espera", "un momento un momento",
            "déjame hablar", "te interrumpo", "perdón que te interrumpa",
            "antes de que sigas"
        ],
        
        # Dudas y vacilaciones
        "Proyecta más seguridad en tu comunicación": [
            "no estoy seguro", "tal vez", "quizás",
            "no sé si", "puede que", "a lo mejor",
            "no te prometo nada", "voy a intentar",
            "haré lo posible"
        ]
    }

def analize_transcription(file_path, diccionario_consejos):
    """
    Analiza una transcripción y devuelve el consejo más relevante basado en las expresiones encontradas.
    
    Args:
        file_path (str): Ruta del archivo con la transcripción a analizar
        diccionario_consejos (dict): Diccionario con patrones y consejos
    
    Returns:
        dict: Diccionario con el consejo más relevante como clave y las expresiones encontradas como valor
    """
    # Leer el archivo con el manejo de codificación mejorado
    transcripcion = read_file_with_fallback_encoding(file_path)
    texto = transcripcion.lower()
    
    best_count = 0
    best_consejo = None
    best_expresiones = []
    
    for consejo, expresiones in diccionario_consejos.items():
        expresiones_encontradas = []
        for expresion in expresiones:
            if expresion.lower() in texto:
                expresiones_encontradas.append(expresion)
        
        if expresiones_encontradas:
            count = len(expresiones_encontradas)
            if count > best_count:
                best_count = count
                best_consejo = consejo
                best_expresiones = expresiones_encontradas
    if best_consejo:
        return best_consejo
    else: 
        return "No hay consejo disponible"
