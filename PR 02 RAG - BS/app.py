import streamlit as st
import tempfile
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM


@st.cache_resource
def load_embeddings():
    """Loads the sentence-transformer embeddings model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm_pipeline():
    """
    Loads a local LLM using HuggingFacePipeline.
    This function is configured to run a model locally, which aligns with your
    dependencies (torch, transformers). It's generally faster for deployed apps
    and avoids API key dependencies.
    """
    model_id = "google/flan-t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=512
    )

    return HuggingFacePipeline(pipeline=pipe)



def process_pdf(uploaded_file):
    """
    Processes the uploaded PDF file, creates a RAG chain, and returns it.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        semantic_splitter = SemanticChunker(
            embeddings=st.session_state.embeddings,
            breakpoint_threshold_type="percentile",
            min_chunk_size=200, 
            add_start_index=True
        )
        docs = semantic_splitter.split_documents(documents)

        vector_db = Chroma.from_documents(
            documents=docs, 
            embedding=st.session_state.embeddings
        )

        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer concise.

        Context: {context}

        Question: {question}

        Helpful Answer:"""
        
        prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | st.session_state.llm
            | StrOutputParser()
        )

    finally:
        os.unlink(tmp_file_path)

    return rag_chain, len(docs)



def main():
    """Main function to run the Streamlit app."""
    
    st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
    st.title("📄 PDF RAG Assistant")

    st.markdown("""
    **Chat with any PDF document!**

    **How to use:**
    1.  Models will load automatically below.
    2.  **Upload a PDF file** and click the "Process PDF" button.
    3.  **Ask any question** about the document in the text box.
    ---
    """)

    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
        
    if not st.session_state.models_loaded:
        with st.spinner("Loading models... This may take a moment."):
            st.session_state.embeddings = load_embeddings()
            st.session_state.llm = load_llm_pipeline()
            st.session_state.models_loaded = True
        st.success("✅ Models loaded successfully!")

    uploaded_file = st.file_uploader("Upload your PDF file", type='pdf')

    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF... Creating chunks and vector store."):
                st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
                st.success(f'✅ PDF processed successfully! Created {num_chunks} chunks.')
                st.info("You can now ask questions about the document below.")

    if st.session_state.rag_chain:
        question = st.text_input(
            "Ask a question about the PDF:",
            placeholder="What is the main topic of this document?"
        )

        if question:
            with st.spinner("Thinking..."):
                output = st.session_state.rag_chain.invoke(question)
                st.write("#### Answer:")
                st.write(output)

if __name__ == "__main__":
    main()
