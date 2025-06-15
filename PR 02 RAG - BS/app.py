import streamlit as st
import tempfile
import os 
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub

from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.bfloat16, 
        low_cpu_mem_usage = True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id = tokenizer.eos_token_id,
        device_map = 'auto'
    )

    return HuggingFacePipeline(pipeline=model_pipline)




# func for pdf
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # save file temp

    # read file 
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # semantic chunking
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=  1, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_size_chunk = 500,
        add_start_index=True
    )


    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)

    retriever = vector_db.as_retriever() # get db in this doc for searching !

    prompt = hub.pull("rlm/rag-prompt") # a sample prompt like: Base on context: {context}
    # and answer this question: {question}

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context":retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    os.unlink(tmp_file_path) # clean temp file after run

    return rag_chain, len(docs)





if __name__  == "__main__":
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "llm" not in st.session_state:
        st.session_state.llm = None

    # buid UI
    st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
    st.title("PDF RAG Assistant")

    st.markdown("""
    **AI Application helps you can chat with PDF file**
    **How to use ? ** 
    1. **Upload file PDF** Upload your file then press Process
    2. **Ask any question** Ask
    --- 
    """)

    #load model

    if not st.session_state.models_loaded:
        st.info("Loading models ...")
        st.session_state.embeddings = load_embeddings()

        st.session_state.llm = load_llm()
        st.session_state.model_loaded = True
        st.success("Models ready ")


    # upload file and process file pdf
    uploaded_file = st.file_uploader("Upload file PDF", type ='pdf')
    if uploaded_file and st.button("Process file PDF"):
        with st.spinner("Processing"):
            st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)

            st.success(f'Complete ! {num_chunks} chunks')
        

    # UI QA
    if st.session_state.rag_chain:
        question = st.text_input("Question:")

        if question:
            with st.spinner("Replying"):
                output = st.session_state.rag_chain.invoke(question)
                answer = output.split("Answer:")[1].strip() if "Anwser:" in output else output.strip()
            
                st.write("**Answer: **")
                st.write(answer)

        