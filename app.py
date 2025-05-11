import os
import tempfile
import logging
import warnings
import streamlit as st
from dotenv import load_dotenv

# LangChain components
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Set page config
st.set_page_config(page_title="NormaBot", page_icon="🤖")
st.title("🧠 Ask Your PDFs - NormaBot")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def process_pdfs(uploaded_files):
    """Process uploaded PDF files and create vector store"""
    documents = []
    
    # Store files in session state for processing
    st.session_state.uploaded_files = uploaded_files
    
    # Process each PDF file
    for pdf_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_file_path)
            documents.extend(loader.load())
        finally:
            os.unlink(tmp_file_path)  # Cleanup temp file

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(documents)
    
    # Create vector store
    index_creator = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=text_splitter
    )
    return index_creator.from_documents(split_documents).vectorstore

def get_qa_chain(vectorstore):
    """Create RetrievalQA chain with Groq/Llama3"""
    prompt_template = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Answer the question based only on the following context:
        {context}
        
        Question: {question}
        Answer directly and precisely without introductory text."""
    )
    
    return RetrievalQA.from_chain_type(
        llm=ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

# Sidebar for PDF upload
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Process PDFs") and uploaded_files:
        with st.spinner("Analyzing documents..."):
            try:
                st.session_state.vectorstore = process_pdfs(uploaded_files)
                st.success(f"Processed {len(uploaded_files)} documents!")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle user input
prompt = st.chat_input("Ask about your documents")
if prompt:
    # Add user message to chat history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if not st.session_state.vectorstore:
        st.error("Please upload and process documents first!")
        st.stop()
    
    try:
        # Generate response
        qa_chain = get_qa_chain(st.session_state.vectorstore)
        result = qa_chain({"query": prompt})
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(result["result"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["result"]
            })
            
            # Show sources
            with st.expander("Document Sources"):
                for doc in result["source_documents"]:
                    source = os.path.basename(doc.metadata["source"])
                    page = doc.metadata.get("page", "N/A")
                    st.write(f"📄 {source} (Page {page})")
                    st.caption(doc.page_content[:300] + "...")
                    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")