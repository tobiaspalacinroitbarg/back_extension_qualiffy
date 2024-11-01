# Librerías
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from funcs import transcribe_diarized, generate_ollama, dict_consejos, analize_transcription, calculate_word_percentage
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("/media", exist_ok=True)

@app.post("/process-mc")
async def process_audio(file: UploadFile = File(...)):
    try:
        logger.info("Iniciando procesamiento de manejo y consejo")
        
        # Guardar el archivo de audio
        file_path = f"media/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info("Audio guardado, iniciando transcripción")
        
        # Transcribir el audio
        transcribe_diarized(
            path=file_path,
            num_speakers=2,
            language="Spanish",
            model_size="base"
        )
        
        logger.info("Transcripción completada, calculando métricas")
        
        transcript_path = "media/transcript.txt"

        # Calcular porcentaje de palabras
        manejo = calculate_word_percentage(transcript_path, "SPEAKER 1")
        
        logger.info(f"Métricas calculadas: {manejo}%, generando consejo")
        
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"No se encontró el archivo de transcripción en {transcript_path}")
        
        problema, solucion = generate_ollama(transcript_path, {"Vendedor": "Jeronimo Carrascal","Servicios":"Campañas de LinkedIn"})
        # Calcular consejo
        dicc = dict_consejos()
        consejo = analize_transcription(transcript_path,dicc)
        response = {
            "consejo": str(consejo),
            "manejo": str(manejo),
            "problema":str(problema),
            "solucion":str(solucion)
        }
        logger.info("Proceso completado exitosamente")
        return response
        
    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.post("/process-advice")
async def process_audio(file: UploadFile = File(...)):
    try:
        logger.info("Iniciando generación de consejo")
    
        transcript_path = "media/transcript.txt"
        
        datos_vendedor = {
            "nombre": "Jeronimo Carrascal",
            "rol": "Vendedor",
            "servicios":"Campañas de LinkedIn"
        }
        
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"No se encontró el archivo de transcripción en {transcript_path}")
            
        problema, solucion = generate_ollama(transcript_path, datos_vendedor)
        logger.info("Consejo generado, extrayendo secciones")
    
        response = {
            "problema": str(problema),
            "solucion":str(solucion)
        }
        logger.info("Limpiando archivos temporales")
        logger.info("Proceso completado exitosamente")
        return response
        
    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}", exc_info=True)
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)