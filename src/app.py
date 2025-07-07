import streamlit as st
import os
import time
import torch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# --- Configuration ---
VECTOR_STORE_DIR = 'data/vector_store'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_LOCAL_NAME = "google/flan-t5-base" # Using the smaller, CPU-friendly Flan-T5

# --- Custom Hashing Functions for Streamlit Caching ---
# These functions tell Streamlit how to hash unhashable objects
def hash_hf_embeddings(embedding_obj):
    """Custom hash function for HuggingFaceEmbeddings."""
    # Hash based on the model name and device, which are unique identifiers
    return f"HFEmbeddings-{embedding_obj.model_name}-{embedding_obj.model_kwargs.get('device', 'cpu')}"

def hash_hf_pipeline_llm(llm_obj):
    """Custom hash function for HuggingFacePipeline LLM."""
    # Hash based on the underlying model's name
    return f"HFPipelineLLM-{llm_obj.pipeline.model.config._name_or_path}"

def hash_chroma_vectorstore(vectorstore_obj):
    """Custom hash function for Chroma vector store."""
    # Hash based on the persist directory
    return f"ChromaDB-{vectorstore_obj._persist_directory}"


# --- RAG Core Logic Functions (from Task 3) ---

# Apply hash_funcs to relevant decorators
HF_EMBEDDINGS_HASH_FUNCS = {HuggingFaceEmbeddings: hash_hf_embeddings}
HF_PIPELINE_LLM_HASH_FUNCS = {HuggingFacePipeline: hash_hf_pipeline_llm}
CHROMA_VECTORSTORE_HASH_FUNCS = {Chroma: hash_chroma_vectorstore}


@st.cache_resource(hash_funcs=HF_EMBEDDINGS_HASH_FUNCS) # <--- ADDED hash_funcs
def get_embedding_model(model_name: str):
    """
    Loads a HuggingFace embedding model.
    Forces CPU usage as per user's system configuration (no GPU).
    """
    device = 'cpu'
    st.info(f"Loading embedding model '{model_name}' on device: {device}")
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        st.success(f"Embedding model '{model_name}' loaded successfully.")
        return embeddings_model
    except Exception as e:
        st.error(f"Error loading embedding model {model_name}: {e}")
        st.exception(e) # Display full traceback in Streamlit
        return None

@st.cache_resource(hash_funcs={**HF_EMBEDDINGS_HASH_FUNCS, **CHROMA_VECTORSTORE_HASH_FUNCS}) # <--- ADDED hash_funcs
# Add underscore to embedding_function to tell Streamlit not to hash it directly
def load_vector_store(persist_directory: str, _embedding_function):
    """
    Loads the persisted ChromaDB vector store.
    """
    st.info(f"Loading vector store from '{persist_directory}'...")
    if not os.path.exists(persist_directory):
        st.error(f"Vector store directory '{persist_directory}' not found. Please run Task 2 first.")
        return None
    try:
        # Use the underscored argument in the function body
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=_embedding_function)
        st.success("Vector store loaded successfully.")
        return vectordb
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.exception(e) # Display full traceback in Streamlit
        return None

@st.cache_resource(hash_funcs=HF_PIPELINE_LLM_HASH_FUNCS) # <--- ADDED hash_funcs
def get_local_llm_model(model_name: str):
    """
    Loads a local LLM using HuggingFace Transformers pipeline.
    Uses AutoModelForSeq2SeqLM for T5 models, which are CPU-friendly.
    """
    st.info(f"Loading local LLM model: {model_name} (This should be relatively fast for Flan-T5 on CPU)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" # This will automatically place the model on CPU if no GPU is available
        )

        pipe = pipeline(
            "text2text-generation", # Changed task for T5 models
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # Max tokens for the LLM to generate in response
            repetition_penalty=1.1, # Avoids repetitive text
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        st.success(f"Local LLM '{model_name}' loaded successfully. Device map: {model.hf_device_map}")
        return llm
    except Exception as e:
        st.error(f"Error loading local LLM model {model_name}: {e}")
        st.exception(e) # Display full traceback in Streamlit
        return None

@st.cache_resource(hash_funcs={**CHROMA_VECTORSTORE_HASH_FUNCS, **HF_PIPELINE_LLM_HASH_FUNCS}) # <--- ADDED hash_funcs
def implement_rag_system(_vector_store, _llm): # Underscored arguments
    """
    Implements the Retrieval-Augmented Generation (RAG) system.
    Combines the vector store retriever with the LLM using a specific prompt.
    """
    st.info("Implementing RAG system...")
    # Prompt Template as per Task 3 requirements
    template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

    Context: {context}
    Question: {question}
    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    if _vector_store is None or _llm is None:
        st.error("Error: Vector store or LLM not initialized for RAG system.")
        return None

    try:
        rag_chain = RetrievalQA.from_chain_type(
            llm=_llm,
            retriever=_vector_store.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 relevant documents
            return_source_documents=True, # Important for displaying sources
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        st.success("RAG system implemented successfully.")
        return rag_chain
    except Exception as e:
        st.error(f"Error implementing RAG system: {e}")
        st.exception(e) # Display full traceback in Streamlit
        return None

# --- Function to initialize all RAG components ---
# This function orchestrates the loading of all components and is cached by Streamlit
@st.cache_resource(show_spinner=False, hash_funcs={**HF_EMBEDDINGS_HASH_FUNCS, **CHROMA_VECTORSTORE_HASH_FUNCS, **HF_PIPELINE_LLM_HASH_FUNCS}) # <--- ADDED hash_funcs
def initialize_all_rag_components():
    embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)
    if embeddings is None:
        return None

    # Pass embeddings to load_vector_store
    vectordb = load_vector_store(VECTOR_STORE_DIR, embeddings)
    if vectordb is None:
        return None

    llm_model = get_local_llm_model(LLM_MODEL_LOCAL_NAME)
    if llm_model is None:
        return None

    # Pass both vectordb and llm_model to implement_rag_system
    rag_system = implement_rag_system(vectordb, llm_model)
    return rag_system

# --- Streamlit App Layout ---
st.set_page_config(page_title="CrediTrust Complaint Analysis Chatbot", page_icon="🗣️", layout="centered")

st.markdown(
    """
    <style>
    .reportview-container .main {
        flex-direction: row;
    }
    .stSpinner > div > div {
        border-top-color: #FF4B4B;
        border-left-color: #FF4B4B;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #e6f7ff; /* Light blue for user messages */
        align-self: flex-end;
    }
    .stChatMessage.assistant {
        background-color: #f0f2f6; /* Light gray for assistant messages */
        align-self: flex-start;
    }
    .stExpander {
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("🗣️ CrediTrust Complaint Analysis Chatbot")
st.markdown("Ask questions about financial service customer complaints and get answers with relevant source documents.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

# --- Load RAG System (only once per app run) ---
# Use a top-level spinner for the entire RAG system initialization
with st.spinner("Initializing RAG system (loading models and data)... This might take a moment."):
    if st.session_state.rag_system is None:
        st.session_state.rag_system = initialize_all_rag_components()

rag_system = st.session_state.rag_system

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Show Sources"):
                st.markdown(message["sources"])

# --- User Input and Response Generation ---
if prompt := st.chat_input("Type your question here...", disabled=(rag_system is None)):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for streaming or direct output
        full_response = ""
        full_sources = ""

        if rag_system is None:
            full_response = "RAG system is not ready. Please wait for initialization or check previous errors."
        else:
            with st.spinner("AI is thinking..."):
                start_time = time.time()
                try:
                    result = rag_system.invoke({"query": prompt})
                    response_time = time.time() - start_time

                    full_response = result['result']
                    # Add a note about response time
                    full_response += f"\n\n*(Response Time: {response_time:.2f} seconds)*"

                    sources_list = []
                    if result.get('source_documents'):
                        for i, doc in enumerate(result['source_documents']):
                            source_id = doc.metadata.get('original_complaint_id', 'N/A')
                            product = doc.metadata.get('product', 'N/A')
                            content_preview = doc.page_content[:500].replace('\n', ' ') + '...' # Preview first 500 chars
                            
                            sources_list.append(
                                f"- **Source {i+1} (Complaint ID: {source_id}, Product: {product}):**\n"
                                f"  ```\n{content_preview}\n  ```"
                            )
                        full_sources = "### Source Documents:\n" + "\n".join(sources_list)
                    else:
                        full_sources = "No relevant source documents found in the context."

                except Exception as e:
                    full_response = f"An error occurred while processing your request: {e}"
                    full_sources = "No sources available due to error."
                    st.exception(f"Error during RAG query: {e}")

        # Display the full response and add to chat history
        message_placeholder.markdown(full_response)
        if full_sources:
            with st.expander("Show Sources"):
                st.markdown(full_sources)

        st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": full_sources})

# --- Clear Chat Button ---
def clear_chat_history():
    st.session_state.messages = []
    st.rerun() # Rerun to clear messages from UI

st.sidebar.button('Clear Chat', on_click=clear_chat_history)

# Provide instructions to run
st.sidebar.markdown(
    """
    ## How to Use:
    1.  **Wait for Initialization:** The system will load models and data. This might take a moment, especially on the first run.
    2.  **Ask a Question:** Once "RAG system initialized and ready!" appears, type your question in the input box below.
    3.  **Get Answer:** Press Enter or click the send button.
    4.  **Review Sources:** Click "Show Sources" below the AI's answer to see the complaint excerpts used.
    5.  **Reset:** Use "Clear Chat" in the sidebar to start a new conversation.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**LLM:** {LLM_MODEL_LOCAL_NAME}")
st.sidebar.markdown(f"**Embedding Model:** {EMBEDDING_MODEL_NAME}")
st.sidebar.markdown(f"**Vector Store:** {VECTOR_STORE_DIR}")
st.sidebar.markdown(f"**Device:** {'CPU' if not torch.cuda.is_available() else 'GPU'}")