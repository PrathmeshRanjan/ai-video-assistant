from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=docs,
        collection_name="transcript_vectors",
        embedding=get_embeddings(),
        persist_directory="chroma-db",
    )

    return vector_store