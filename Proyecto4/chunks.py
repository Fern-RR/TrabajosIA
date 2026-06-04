from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Cargar todos los PDFs del directorio
ruta_corpus = "corpus_violencia"
print(f"Cargando documentos desde: {ruta_corpus}...")

loader = PyPDFDirectoryLoader(ruta_corpus)
documentos = loader.load()

print(f"¡Éxito! Se cargaron {len(documentos)} páginas en total de los 15 PDFs.\n")

# 2. Revisar páginas con poco texto
print("Revisando calidad de extracción...")

paginas_con_poco_texto = []

for i, doc in enumerate(documentos):
    texto = doc.page_content.strip()

    if len(texto) < 100:
        paginas_con_poco_texto.append({
            "indice": i,
            "archivo": doc.metadata.get("source"),
            "pagina": doc.metadata.get("page"),
            "caracteres": len(texto)
        })

print(f"Páginas con poco texto extraído: {len(paginas_con_poco_texto)}")

for pagina in paginas_con_poco_texto[:20]:
    print(pagina)

# 3. Filtrar páginas casi vacías
documentos_utiles = [
    doc for doc in documentos
    if len(doc.page_content.strip()) >= 100
]

print(f"\nPáginas útiles para chunking: {len(documentos_utiles)}")

# 4. Configurar el chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# 5. Cortar los documentos
print("\nCortando los documentos en fragmentos...")
chunks = text_splitter.split_documents(documentos_utiles)

print(f"Proceso terminado. Se generaron {len(chunks)} fragmentos de texto.")

# 6. Imprimir una muestra para verificar
if chunks:
    print("\n--- Muestra del primer Chunk ---")
    print(chunks[0].page_content)
    print(f"\nMetadata original: {chunks[0].metadata}")