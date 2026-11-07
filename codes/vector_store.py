
from langchain_chroma import Chroma


def create_vector_store(chunks, embeddings):

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="college_academic_assistant"
    )

    print("ChromaDB vector store created successfully.")

    return vectorstore
