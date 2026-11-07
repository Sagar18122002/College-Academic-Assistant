
def retrieve_documents(question, vectorstore, k=10):
    documents = vectorstore.similarity_search(
        question,
        k=k
    )

    return documents
