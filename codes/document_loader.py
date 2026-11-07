
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Academic_Regulations.pdf"


def load_documents():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DATA_PATH}"
        )

    loader = PyPDFLoader(
        str(DATA_PATH)
    )

    documents = loader.load()

    for doc in documents:
        doc.metadata.update({
            "source": DATA_PATH.name,
            "page": 1,
            "document_type": "academic_policy"
        })

    return documents
