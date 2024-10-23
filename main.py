from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from funcs import transcribe_diarized, generar_consejo_con_ollama

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Asegúrate de que exista el directorio para los archivos
os.makedirs("/media", exist_ok=True)

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Guardar el archivo de audio
        file_path = f"/media/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Transcribir el audio (ajusta los parámetros según necesites)
        transcribe_diarized(
            path=file_path,
            num_speakers=2,  # Asumiendo 2 hablantes: vendedor y cliente
            language="Spanish",
            model_size="base"
        )
        
        # Generar consejo (ajusta los datos del vendedor según tu necesidad)
        datos_vendedor = {
            "nombre": "Jeronimo Carrascal",
            "rol": "Vendedor"
        }
        
        consejo = generar_consejo_con_ollama("/media/transcript.txt", datos_vendedor)
        
        # Limpiar archivos temporales
        os.remove(file_path)
        os.remove("/media/transcript.txt")
        
        return {"consejo": consejo}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)