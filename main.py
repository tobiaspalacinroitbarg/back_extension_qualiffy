from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from funcs import transcribe_diarized, generar_consejo_con_ollama, extract_ia_sections, calculate_word_percentage
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

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        logger.info("Iniciando procesamiento de audio")
        
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
        
        # Generar consejos
        datos_vendedor = {
            "nombre": "Jeronimo Carrascal",
            "rol": "Vendedor"
        }
        
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"No se encontró el archivo de transcripción en {transcript_path}")
            
        consejo = generar_consejo_con_ollama(transcript_path, datos_vendedor)
        logger.info("Consejo generado, extrayendo secciones")
        
        consejos = extract_ia_sections(consejo)
        if not all(consejos):
            logger.warning("Algunas secciones del consejo están vacías")
            
        response = {
            "consejo": consejos[1] if consejos[1] else "No se pudo generar un consejo",
            "asesor": consejos[0] if consejos[0] else "No se pudo generar una evaluación",
            "manejo": str(manejo)
        }
        
        logger.info("Limpiando archivos temporales")
        # Limpiar archivos temporales
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
        if os.path.exists("audio.wav_mono.wav"):
            os.remove("audio.wav_mono.wav")
        
        logger.info("Proceso completado exitosamente")
        return response
        
    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}", exc_info=True)
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)