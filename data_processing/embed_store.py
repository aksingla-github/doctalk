from constants import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_NAME, MODEL_NAME
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import traceback
import pdfplumber

embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)


def read_file_in_chunks(file_path, ext):
    if ext == 'pdf':
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''

    elif ext == 'txt':
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    text_chunks = splitter.split_text(text)
    return text_chunks


def get_embeddings(chunks):
    embeddings = embedding_model.embed_documents(chunks)
    return embeddings


def store_embeddings(embeddings, chunks):
    if os.path.exists(FAISS_INDEX_NAME):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(FAISS_INDEX_NAME, embedding_model, allow_dangerous_deserialization=True)
        vector_store.add_embeddings(list(zip(chunks, embeddings)))
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_embeddings(list(zip(chunks, embeddings)), embedding_model)

    vector_store.save_local(FAISS_INDEX_NAME)

    test_results = vector_store.similarity_search(chunks[0], k=1)
    print(test_results)
    if not test_results:
        raise RuntimeError("FAISS index update verification failed. No results found!")


def embed_and_store(file_path, ext):
    try:
        print("Chunking file content...")
        text_chunks = read_file_in_chunks(file_path, ext)
        print(f"{len(text_chunks)} chunks created.")
        print(f"Converting chunks to embeddings")
        embeddings = get_embeddings(text_chunks)
        print(f"Embeddings created. First vector sample: {embeddings[0][:5]}")
        store_embeddings(embeddings, text_chunks)
    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__ == "__main__":
    embed_and_store('../uploads/elon_musk_info.txt', 'txt')
