from funcs import generate_ollama, transcribe_deepgram, print_transcript,create_transcript

if __name__=="__main__":
    datos_vendedor={"Vendedor":"Jerónimo Carrascal","Servicios":"Campañas de LinkedIn"}
    #transcribe_diarized(path = 'media/audio.wav', num_speakers = 2, language="Spanish", model_size='large')
    #print(calculate_word_percentage("media/transcript.txt","SPEAKER 1"))
    #print(top_words_from_transcript("media/transcript.txt")[0][0])
    transcribe_deepgram("audio.wav")
    create_transcript("audio.json","transcript.txt")
    #problema, solucion = generate_ollama("media/audio.txt",datos_vendedor)
    #print(problema, solucion)
    #print(extract_ia_sections(consejo))

