
from codes.document_loader import load_documents
from codes.text_splitter import split_documents
from codes.embedding_model import get_embedding_model
from codes.vector_store import create_vector_store
from codes.retriever import retrieve_documents
from codes.reranker import get_reranker, rerank_documents
from codes.llm_service import get_answer_chain


def build_rag_pipeline():
    """Initialize the components needed by the RAG application."""

    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = get_embedding_model()
    vectorstore = create_vector_store(chunks, embeddings)

    reranker = get_reranker()
    answer_chain = get_answer_chain()

    return vectorstore, reranker, answer_chain


def answer_question(question, vectorstore, reranker, answer_chain):
    """Retrieve, rerank, and answer a user's question."""

    candidates = retrieve_documents(
        question,
        vectorstore,
        k=10
    )

    ranked_documents = rerank_documents(
        question,
        candidates,
        reranker,
        top_n=3
    )

    if not ranked_documents:
        return (
            "No relevant documents were found.",
            []
        )

    context = "\n\n".join(
        doc.page_content
        for doc, score in ranked_documents
    )

    answer = answer_chain.invoke({
        "question": question,
        "context": context
    })

    return answer, ranked_documents
