from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from constants import FAISS_INDEX_NAME, MODEL_NAME
import os

embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)


def get_embeddings(query):
    embeddings = embedding_model.embed_query(query)
    return embeddings


def format_response(retrieved_chunks, query):
    from langchain.llms import HuggingFacePipeline
    from transformers import pipeline
    pipe = pipeline(
        "text-generation",
        model="microsoft/phi-2",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    template = """
    Given the context below, answer the question precisely and clearly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    context = "\n\n".join(retrieved_chunks)
    # Generate the answer using your local LLM
    response = llm_chain.invoke({
        "context": context,
        "question": query
    })
    return response['text'].strip()


def search_faiss(query, top_k):
    if not os.path.exists(FAISS_INDEX_NAME):
        raise RuntimeError('FAISS index not found. Please process and store embeddings first.')
    vector_store = FAISS.load_local(FAISS_INDEX_NAME, embedding_model, allow_dangerous_deserialization=True)
    query_vector = get_embeddings(query)
    print(len(query_vector))
    results = vector_store.similarity_search_by_vector(query_vector, k=top_k)

    retrieved_chunks = [doc.page_content for doc in results]
    # response = format_response(retrieved_chunks, query)
    response = '\n'.join(retrieved_chunks)
    print(response)
    return response


if __name__ == "__main__":
    search_faiss('who is elon musk', 3)
