
from sentence_transformers import CrossEncoder


def get_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def rerank_documents(question, documents, reranker, top_n=3):

    if not documents:
        return []

    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return ranked_documents[:top_n]
