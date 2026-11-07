
import time
import streamlit as st

from codes.rag_pipeline import build_rag_pipeline, answer_question


st.set_page_config(
    page_title="College Academic Assistant",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def initialize_pipeline():
    return build_rag_pipeline()


st.title("🎓 College Academic Assistant")
st.write(
    "Ask questions about academic policies, examinations, "
    "assignments, library rules, and placements."
)

st.warning(
    "Demo only: The current policy document contains sample "
    "information. Verify rules against official college documents."
)

with st.sidebar:
    st.header("Project Details")
    st.write("**Architecture:** RAG")
    st.write("**Vector store:** ChromaDB")
    st.write("**Reranking:** Cross-Encoder")
    st.write("**LLM:** Gemini")
    st.write("**UI:** Streamlit")

question = st.text_area(
    "Enter your question",
    placeholder="What attendance percentage is required?",
    height=100
)

if st.button("🔎 Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Initializing RAG components..."):
                vectorstore, reranker, answer_chain = (
                    initialize_pipeline()
                )

            start_time = time.perf_counter()

            with st.spinner("Retrieving and analyzing documents..."):
                answer, ranked_documents = answer_question(
                    question,
                    vectorstore,
                    reranker,
                    answer_chain
                )

            elapsed = time.perf_counter() - start_time

            st.subheader("✅ Answer")
            st.markdown(answer)

            st.caption(f"Processing time: {elapsed:.2f} seconds")

            if ranked_documents:
                st.subheader("📚 Retrieved Sources")

                for index, (doc, score) in enumerate(
                    ranked_documents, start=1
                ):
                    source = doc.metadata.get(
                        "source", "Unknown"
                    )
                    page = doc.metadata.get(
                        "page", "Not available"
                    )

                    with st.expander(
                        f"Source {index}: {source}"
                    ):
                        st.write(f"**Page:** {page}")
                        st.write(
                            f"**Document type:** "
                            f"{doc.metadata.get('document_type', 'Unknown')}"
                        )
                        st.write(
                            f"**Reranker score:** {float(score):.4f}"
                        )
                        st.write("**Retrieved text:**")
                        st.write(doc.page_content)

        except Exception as error:
            st.error("The application encountered an error.")
            st.exception(error)
