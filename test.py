from funcs import transcribe_diarized, generar_consejo_con_ollama

if __name__=="__main__":
    datos_vendedor={"Nombre completo":"Manuel Perez","Servicios":["Asistente con IA en tiempo real en llamadas de Google Meet para aumentar las ventas","Seguimientos inteligentes con IA que se acoplan con el calnedario de Google."],"Pdfs":[]}
    transcribe_diarized(path = 'media/audio.wav', num_speakers = 2, language="Spanish", model_size='large')
    print(generar_consejo_con_ollama("media/transcript.txt",datos_vendedor))
