# Librerías
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from funcs import transcribe_diarized, generate_ollama, dict_consejos, analize_transcription, calculate_word_percentage
import logging
from datetime import datetime
import json
from pathlib import Path
import tempfile

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

# Crear directorios necesarios
os.makedirs("media", exist_ok=True)
os.makedirs("media/chunks", exist_ok=True)
os.makedirs("media/transcriptions", exist_ok=True)

# Estado global para manejar las sesiones de grabación
active_sessions = {}

class RecordingSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.chunk_count = 0
        self.transcriptions = []
        self.total_words_speaker1 = 0
        self.total_words_speaker2 = 0
        self.session_dir_chunks = Path(f"media/chunks/{session_id}")
        self.session_dir_chunks.mkdir(exist_ok=True)  
        self.session_dir_transcriptions = Path(f"media/transcriptions/{session_id}")
        self.session_dir_transcriptions.mkdir(exist_ok=True)
        
    def add_transcription(self, transcription):
        self.transcriptions.append(transcription)
        
    def get_combined_transcription(self):
        return "\n".join(self.transcriptions)
    
    def update_word_counts(self, speaker1_words, speaker2_words):
        self.total_words_speaker1 += speaker1_words
        self.total_words_speaker2 += speaker2_words
        
    def calculate_current_ratio(self):
        total = self.total_words_speaker1 + self.total_words_speaker2
        if total == 0:
            return 0
        return (self.total_words_speaker1 / total) * 100

@app.post("/start-session")
async def start_session():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    active_sessions[session_id] = RecordingSession(session_id)
    return {"session_id": session_id}

@app.post("/process-chunk/{session_id}")
async def process_chunk(session_id: str, file: UploadFile = File(...)):
    try:
        if session_id not in active_sessions:
            raise ValueError("Sesión no encontrada")
            
        session = active_sessions[session_id]
        session.chunk_count += 1
        
        # Guardar el chunk de audio
        chunk_path = session.session_dir_chunks / f"chunk_{session.chunk_count}.wav"
        transcript_path = session.session_dir_transcriptions / f"chunk_{session.chunk_count}.txt"
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Procesando chunk {session.chunk_count} de sesión {session_id}, transcript_path:{transcript_path}")
        
        # Transcribir el chunk
        transcription = transcribe_diarized(
            path=str(chunk_path),
            num_speakers=2,
            language="es",
            model_size="base",
            output_path = transcript_path
        )
        
        # Guardar transcripción
        session.add_transcription(transcription)
        
        
        # Actualizar conteos de palabras
        speaker1_words = len([word for line in transcription.split('\n') 
                            if line.startswith('SPEAKER 1') 
                            for word in line.split()[2:]])
        speaker2_words = len([word for line in transcription.split('\n') 
                            if line.startswith('SPEAKER 2') 
                            for word in line.split()[2:]])
        session.update_word_counts(speaker1_words, speaker2_words)
        
        # Calcular métricas actuales
        current_manejo = session.calculate_current_ratio()
        
        # Generar consejo basado en la transcripción completa
        dicc = dict_consejos()
        consejo = analize_transcription(str(transcript_path), dicc)
        
        #os.unlink(str(transcript_path))
        
        response = {
            "chunk_number": session.chunk_count,
            "transcription": transcription,
            "consejo": consejo,
            "manejo": round(current_manejo, 1)
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error en el procesamiento del chunk: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.post("/end-session/{session_id}")
async def end_session(session_id: str):
    try:
        if session_id not in active_sessions:
            raise ValueError("Sesión no encontrada")
            
        session = active_sessions[session_id]
        
        # Combinar todas las transcripciones
        final_transcription = session.get_combined_transcription()
        
        # Guardar transcripción final
        transcription_path = Path(f"media/transcriptions/{session_id}_final.txt")
        transcription_path.write_text(final_transcription)
        
        # Calcular métricas finales
        final_manejo = session.calculate_current_ratio()
        
        # Generar consejo final
        dicc = dict_consejos()
        final_consejo = analize_transcription(str(transcription_path), dicc)
        
        # Limpiar recursos
        # shutil.rmtree(session.session_dir_chunks)  # Descomenta si quieres eliminar los chunks
        del active_sessions[session_id]
        
        return {
            "transcription_path": str(transcription_path),
            "final_manejo": round(final_manejo, 1),
            "final_consejo": final_consejo
        }
        
    except Exception as e:
        logger.error(f"Error al finalizar la sesión: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    if session_id not in active_sessions:
        return {"error": "Sesión no encontrada"}
    
    session = active_sessions[session_id]
    return {
        "chunk_count": session.chunk_count,
        "current_manejo": round(session.calculate_current_ratio(), 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)