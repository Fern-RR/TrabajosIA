import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Cargar todos los PDFs del directorio
ruta_corpus = "corpus_violencia"
print(f"Cargando documentos desde: {ruta_corpus}...")

loader = PyPDFDirectoryLoader(ruta_corpus)
documentos = loader.load()

print(f"Se cargaron {len(documentos)} páginas en total.\n")

# 2. Filtrar páginas con muy poco texto
documentos_utiles = [
    doc for doc in documentos
    if len(doc.page_content.strip()) >= 100
]

print(f"Páginas útiles para chunking: {len(documentos_utiles)}")

# 3. Configurar chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(documentos_utiles)

print(f"Se generaron {len(chunks)} fragmentos de texto.\n")

# 4. Configurar modelo de embeddings local
modelo_nombre = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"Cargando modelo de embeddings: {modelo_nombre}...")

embeddings = HuggingFaceEmbeddings(
    model_name=modelo_nombre
)

# 5. Crear y guardar base vectorial
directorio_db = "chroma_db_violencia"

print(f"Convirtiendo texto a vectores y guardando en: {directorio_db}")
print("Esto puede tardar unos minutos...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=directorio_db,
    collection_name="violencia_mexico"
)

print("\n¡Éxito! La base de datos vectorial fue creada y guardada localmente.")

