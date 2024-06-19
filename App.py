import os
import pathlib
import textwrap
from dotenv import load_dotenv
from xml.dom.minidom import CharacterData

import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings , GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
    
def main ():
    
    load_dotenv() 
    
    st.set_page_config(page_title="Chat con multiples PDFs" , page_icon=":books:")    
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:        
        st.session_state.chat_history = []
        
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Pregunta lo que desees"}
        ]
        
#------------------ Header    
    st.header("Chatea con tus PDFs")
    
    pregunta = st.text_input("Pregunta lo que quieras sobre los temas en tus PDFs")    
    
    if pregunta :
        maneja_pregunta( pregunta )
    
#------------------ Sidebar
    with st.sidebar:
        st.subheader("Tus documentos")
        
        pdf_docs = st.file_uploader( "Sube tus PDFs y clickea en -Procesar-", accept_multiple_files=True )
        
        if st.button("Procesar") :
            with st.spinner ("Procesando"):
                
                # Toma el texto
                texto_crudo = texto_pdf(pdf_docs)
                
                 # Toma los fragmentos o "chunks"
                chunks_texto = extrae_chunks(texto_crudo)
                
                # Crea vectores
                base_vectores = crear_vectores(chunks_texto)
                
                # Crea cadena de conversacion
                st.session_state.conversation = crea_cadena_conversacion(base_vectores)             
    
#-------------------------------------------------------------------------------------------------------------------- 
#-------------------------------------------- Toma el texto de los PDFs            
def texto_pdf(pdf_docs):
    texto = ""
    for pdf in pdf_docs:
        
        lector_pdf = PdfReader(pdf)
        for page in lector_pdf.pages:
            texto += page.extract_text()
            
    return texto

#-------------------------------------------- Separa el texto en partes
def extrae_chunks(texto):
    corta_texto = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    
    chunks = corta_texto.split_text(texto)    
    return chunks

#-------------------------------------------- Crea vectores
def crear_vectores(chunk_texto):
    try:
        
        modelo_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        base_vectores = FAISS.from_texts(texts=chunk_texto, embedding=modelo_embedding)     
        return base_vectores
    
    except Exception as e:
        print(f"Error: {e}")     

#-------------------------------------------- Crea cadena de conversacion
def crea_cadena_conversacion(base_vectores):
    
    api_key = os.environ['GOOGLE_API_KEY']
    llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key)
    
    memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True )
    
    cadena_conversacion = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=base_vectores.as_retriever(),
        memory=memory
    )
    return cadena_conversacion

#-------------------------------------------- Maneja preguntas                
def maneja_pregunta(pregunta ):
    
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    respuesta = st.session_state.conversation({'question': pregunta})
    
    with st.chat_message("assistant"):
        st.markdown(respuesta['answer'])        
        
    st.session_state.chat_history.append(
        {
            "role":"user",
            "content": pregunta
        }
    )
    st.session_state.chat_history.append(
        {
            "role":"assistant",
            "content": respuesta
        }
    )
    
#--------------------------------------------------------------------------------------------------------------------     

if __name__ == '__main__':    
    main()