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

def transcribe_diarized(path:str,num_speakers:int, language:str, model_size:str,output_path:str):
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
    try:
        # Agrupamiento para asignar oradores
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
    except:
         # Agrupamiento para asignar oradores
        clustering = AgglomerativeClustering(1).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
    # Función para convertir tiempo en formato legible
    def time_format(secs):
        return str(datetime.timedelta(seconds=round(secs)))

    # Modificar la parte donde se guarda el archivo
    with open(output_path, "w", encoding="utf-8", errors='ignore') as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + time_format(segment["start"]) + '\n')
            f.write(segment["text"][1:] + ' ')

    transcript = read_file_with_fallback_encoding(output_path)

    # Fin del timer y mostrar tiempo de ejecución
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"El tiempo total de transcripción y diarización fue de {execution_time:.2f} segundos.")
    return transcript

def dict_consejos():
    """
    Diccionario expandido de expresiones comunes en llamadas formales de venta que podrían mejorarse,
    incluyendo variantes regionales en español profesional.
    """
    return {
        # Frases débiles al presentarse
        "Fortalece tu introducción inicial": [
            # General/España
            "le molesto para", "interrumpo para", "disculpe que",
            "un momentito", "buenos días", "le llamo de",
            "perdone la molestia", "siento molestar", "espero no interrumpir",
            "si me permite", "quisiera comentarle", "si puedo",
            "llamaba por", "estaba llamando", "le contacto",
            "me preguntaba si", "quería saber si", "podría ser",
            "si tiene tiempo", "un minuto", "brevemente",
            "rápidamente quisiera", "si me escucha", "no sé si",
            "le explico", "déjeme contarle", "tengo entendido",
            "según veo", "por lo visto", "me dijeron",
            
            # Latinoamérica
            "mi nombre es", "le hablo de", "quisiera hablarle",
            "para comentarle", "mire señor", "oiga señora",
            "le hablo para", "quería ver si", "estoy llamando",
            "me comunico para", "le contacto por", "quisiera saber",
            "necesitaba confirmar", "deseaba consultar", "me gustaría ver",
            "podría decirme", "sería tan amable", "si fuera posible",
            "permítame explicarle", "si me permite", "con permiso",
            "disculpe la hora", "si está ocupado", "cuando pueda"
        ],

        # Frases débiles al presentar beneficios
        "Comunica beneficios con más impacto": [
            # General/España
            "más o menos", "aproximadamente", "podría ser",
            "quizás le interese", "tal vez necesite", "puede servirle",
            "le podría funcionar", "sería útil", "vendría bien",
            "ayudaría a", "mejoraría su", "facilitaría",
            "haría que", "permitiría", "posiblemente",
            "en teoría", "según entiendo", "por lo general",
            "normalmente", "usualmente", "típicamente",
            "en algunos casos", "a veces", "dependiendo",
            "si todo va bien", "en principio", "inicialmente",
            
            # Latinoamérica
            "le puede servir", "le podría ayudar", "es como",
            "digamos que", "más que todo", "básicamente es",
            "viene siendo", "sería como", "algo así",
            "similar a", "parecido a", "tipo como",
            "una especie de", "una suerte de", "se asemeja",
            "funciona tipo", "opera como", "trabaja así",
            "sirve para", "ayuda con", "mejora en",
            "hace que", "permite que", "facilita que"
        ],

        # Frases débiles al manejar precio
        "Comunica el precio con seguridad": [
            # General/España
            "sale en", "cuesta como", "está en",
            "le sale", "le costaría", "el precio es",
            "valdría", "tendría valor", "el costo es",
            "la inversión es", "el monto es", "la cantidad es",
            "el total sería", "aproximadamente", "más o menos",
            "por ahí", "alrededor de", "cerca de",
            "ronda los", "oscila entre", "fluctúa en",
            "puede variar", "depende de", "según el caso",
            
            # Latinoamérica
            "le sale en", "viene costando", "estaría en",
            "sale como", "le queda en", "precio aproximado",
            "vendría siendo", "quedaría en", "saldría por",
            "le costaría", "el valor es", "la tarifa es",
            "el pago sería", "el importe es", "la suma es",
            "el total es", "el precio final", "el costo total",
            "la inversión total", "el monto final", "el valor total"
        ],

        # Frases débiles al manejar objeciones
        "Maneja objeciones con más seguridad": [
            # General/España
            "entiendo que", "comprendo que", "si, pero",
            "tiene razón", "claro, aunque", "es verdad",
            "lo entiendo", "ya veo", "me hago cargo",
            "sí, ciertamente", "efectivamente", "naturalmente",
            "por supuesto", "desde luego", "sin duda",
            "en efecto", "evidentemente", "claramente",
            "comprensible", "lógicamente", "obviamente",
            "en realidad", "de hecho", "ciertamente",
            
            # Latinoamérica
            "sí, entiendo", "claro, pero", "le entiendo",
            "puede ser", "es cierto", "tiene razón",
            "comprendo su punto", "veo su preocupación", "entiendo su posición",
            "es comprensible", "es razonable", "es lógico",
            "tiene sentido", "es válido", "es importante",
            "es considerable", "es relevante", "es significativo",
            "me imagino", "supongo que", "asumo que",
            "probablemente", "posiblemente", "seguramente"
        ],

        # Frases débiles al cerrar
        "Fortalece tus cierres": [
            # General/España
            "si le parece", "cuando guste", "si le interesa",
            "piénselo", "consúltelo", "me avisa",
            "podríamos proceder", "seguimos adelante", "avanzamos con",
            "lo tramitamos", "iniciamos el", "comenzamos con",
            "formalizamos", "concretamos", "cerramos",
            "definimos", "acordamos", "establecemos",
            "coordinamos", "programamos", "agendamos",
            "reservamos", "apartamos", "separamos",
            
            # Latinoamérica
            "lo puede pensar", "me dice entonces", "me confirma",
            "me avisa cualquier", "lo consultamos", "nos mantenemos",
            "podemos continuar", "seguimos con", "procedemos con",
            "iniciamos el", "empezamos con", "arrancamos con",
            "formalizamos ya", "concretamos hoy", "cerramos ahora",
            "definimos esto", "acordamos esto", "establecemos ya",
            "coordinamos ahora", "programamos ya", "agendamos esto"
        ],

        # Frases débiles de seguimiento
        "Mejora el seguimiento": [
            # General/España
            "le vuelvo llamar", "lo contacto", "le escribo",
            "nos comunicamos", "le marco", "estamos hablando",
            "me comunico", "le contacto", "vuelvo a llamar",
            "retomo contacto", "me pongo en", "vuelvo con",
            "regreso con", "insisto", "retomo",
            "verifico", "confirmo", "reviso",
            "chequeo", "consulto", "averiguo",
            
            # Latinoamérica
            "lo vuelvo llamar", "le contacto", "nos hablamos",
            "le marco después", "lo busco", "le escribo",
            "me comunico luego", "le contacto después", "vuelvo a llamarlo",
            "retomo contacto", "me pongo en", "vuelvo con",
            "regreso con", "insisto luego", "retomo después",
            "verifico luego", "confirmo después", "reviso y llamo",
            "chequeo y aviso", "consulto y llamo", "averiguo y contacto"
        ],

        # Muletillas formales a evitar
        "Elimina muletillas formales": [
            # General/España
            "este...", "eh...", "mire...",
            "o sea", "digamos", "básicamente",
            "en realidad", "de hecho", "sinceramente",
            "honestamente", "francamente", "ciertamente",
            "efectivamente", "evidentemente", "obviamente",
            "claramente", "simplemente", "prácticamente",
            "teóricamente", "técnicamente", "realmente",
            "verdaderamente", "seguramente", "probablemente",
            
            # Latinoamérica
            "verdad?", "me entiende?", "correcto?",
            "cierto?", "no?", "ok?",
            "me explico?", "me sigue?", "está claro?",
            "me comprende?", "me capta?", "me entiende?",
            "lo tiene?", "lo ve?", "lo nota?",
            "lo percibe?", "lo observa?", "lo aprecia?",
            "se da cuenta?", "lo visualiza?", "lo considera?"
        ],

        # Frases dubitativas
        "Proyecta más seguridad": [
            # General/España
            "no sé si", "tal vez", "posiblemente",
            "quizás", "probablemente", "seguramente",
            "puede que", "podría ser", "sería posible",
            "cabría la", "existiría la", "habría la",
            "se podría", "se debería", "convendría",
            "sugeriría", "propondría", "plantearía",
            "consideraría", "evaluaría", "analizaría",
            
            # Latinoamérica
            "puede ser", "más o menos", "aproximadamente",
            "como que", "por ahí", "alrededor de",
            "tal vez podríamos", "quizás deberíamos", "posiblemente sea",
            "probablemente convenga", "seguramente sirva", "puede que funcione",
            "podría resultar", "sería conveniente", "cabría considerar",
            "existiría posibilidad", "habría chance", "se podría intentar"
        ],

        # Minimizadores
        "Evita minimizar tu propuesta": [
            # General/España
            "solamente para", "nada más para", "solo quería",
            "únicamente", "brevemente", "rápidamente",
            "un momentito", "un segundito", "un instante",
            "pequeño favor", "simple consulta", "breve pregunta",
            "rápida consulta", "corta llamada", "pequeña duda",
            "mínima consulta", "ligera pregunta", "básica duda",
            "sencilla pregunta", "simple duda", "fácil consulta",
            
            # Latinoamérica
            "solo para", "nomás para", "rapidito para",
            "un momento", "un segundito", "un ratito",
            "pequeña consulta", "breve momento", "rápida pregunta",
            "simple duda", "corto instante", "pequeño espacio",
            "mínimo tiempo", "ligero momento", "básica consulta",
            "sencilla duda", "fácil pregunta", "simple momento"
        ],

        # Frases de sumisión
        "Mantén una posición de igual a igual": [
            # General/España
            "si me permite", "si no le molesta", "cuando pueda",
            "si tiene tiempo", "si gusta", "si desea",
            "si lo considera", "si lo estima", "si lo cree",
            "si le parece", "si lo ve", "si lo prefiere",
            "si lo autoriza", "si lo aprueba", "si lo acepta",
            "si me concede", "si me otorga", "si me faculta",
            "si me posibilita", "si me facilita", "si me permite",
            
            # Latinoamérica
            "si me deja", "si me acepta", "si tiene oportunidad",
            "si me autoriza", "si me concede", "si me permite",
            "si lo considera", "si lo permite", "si es posible",
            "si no es molestia", "si no es problema", "si no es inconveniente",
            "si tiene chance", "si puede ser", "si es factible",
            "si es viable", "si es realizable", "si es ejecutable"
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
        print(best_expresiones)
        return best_consejo
    else: 
        return "No hay consejo disponible"
