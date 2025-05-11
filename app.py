import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Configuration
load_dotenv()
st.set_page_config(page_title="Single Doc Chatbot", page_icon="📄")
st.title("📄 Document Chatbot")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

@st.cache_resource
def process_pdf(pdf_file):
    """Process PDF and return vectorstore"""
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process PDF
        loader = PyPDFLoader(tmp_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Create in-memory vectorstore
        index_creator = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
            text_splitter=text_splitter
        )
        
        return index_creator.from_loaders([loader]).vectorstore
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Process PDF button
if uploaded_file and st.button("Process Document"):
    with st.spinner("Analyzing document..."):
        try:
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle queries
if prompt := st.chat_input("Ask about the document"):
    if not st.session_state.vectorstore:
        st.error("Please upload and process a document first!")
        st.stop()
    
    # Add user message to history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama3-8b-8192"
            ),
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain({"query": prompt})
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(result["result"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["result"]
            })
            
            # Show sources
            with st.expander("Source References"):
                for doc in result["source_documents"]:
                    page = doc.metadata.get('page', 'N/A')
                    st.write(f"📄 Page {page}")
                    st.caption(doc.page_content[:300] + "...")
                    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")