from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Original documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    return chunks