import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# App title and presentation
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Visual: fondo negro, texto blanco y fuente Roboto
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainContainer"] {
        background-color: #000000 !important; /* fondo negro */
    }
    .stApp, .css-1outpf7, .css-2trqyj, .stText, .stMarkdown, .stButton, .stHeader, .stSubheader {
        color: #FFFFFF !important; /* texto blanco */
        font-family: 'Roboto', sans-serif !important; /* tipografía Roboto */
    }
    /* Ajustes extra para títulos y áreas de texto */
    h1, h2, h3, h4, h5, h6, .stMarkdown p {
        color: #FFFFFF !important;
        font-family: 'Roboto', sans-serif !important;
    }
    /* Asegurar contraste en inputs y botones */
    .stTextInput > div > input, .stTextArea > div > textarea, .stFileUploader, .stButton button {
        background-color: rgba(255,255,255,0.04) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load and display image: usar la imagen proporcionada llamada 'imagen_robot.png'
try:
    if os.path.exists("imagen_robot.png"):
        image = Image.open("imagen_robot.png")
    else:
        # fallback al nombre original si no existe la imagen nueva
        image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# PDF uploader
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use a current model instead of deprecated text-davinci-003
            # Options: "gpt-3.5-turbo-instruct" or "gpt-4-turbo-preview" depending on your API access
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
