import streamlit as st
import os
import time
import torch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline

# --- Configuration (Ensure these match your previous tasks) ---
VECTOR_STORE_DIR = 'data/vector_store'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_LOCAL_NAME = "HuggingFaceH4/zephyr-7b-beta" # Or your chosen local LLM

# --- Function to initialize the RAG system (cached) ---
@st.cache_resource
def initialize_rag_system():
    """
    Initializes and caches the RAG system components.
    Uses st.cache_resource to load these heavy components only once.
    """
    with st.spinner("Loading embedding model..."):
        embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)
        if embeddings is None:
            st.error("Failed to load embedding model. Check console for details.")
            return None

    with st.spinner("Loading vector store..."):
        vectordb = load_vector_store(VECTOR_STORE_DIR, embeddings)
        if vectordb is None:
            st.error("Failed to load vector store. Please run Task 2 first.")
            return None

    with st.spinner("Loading local LLM model... This may take several minutes..."):
        llm_model = get_local_llm_model(LLM_MODEL_LOCAL_NAME)
        if llm_model is None:
            st.error("Failed to load local LLM model. Check console for details and ensure all dependencies are met (torch, bitsandbytes, accelerate).")
            return None

    with st.spinner("Implementing RAG system..."):
        rag_chain = implement_rag_system(vectordb, llm_model)
        if rag_chain is None:
            st.error("Failed to implement RAG system. Check console for details.")
            return None

    st.success("RAG system initialized and ready!")
    return rag_chain

# --- Loading Functions (Copied from src/rag_system_implementation.py, ensure they are correct) ---
def get_embedding_model(model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.info(f"Using device for embeddings: {device}")
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        return embeddings_model
    except Exception as e:
        st.exception(f"Error loading embedding model {model_name}: {e}")
        return None

def load_vector_store(persist_directory, embedding_function):
    if not os.path.exists(persist_directory):
        st.error(f"Vector store directory '{persist_directory}' not found. Please run Task 2 first.")
        return None
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        return vectordb
    except Exception as e:
        st.exception(f"Error loading vector store: {e}")
        return None

def get_local_llm_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define the quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        st.info(f"Local LLM '{model_name}' loaded successfully. Device map: {model.hf_device_map}")
        return llm
    except Exception as e:
        st.exception(f"Error loading local LLM model {model_name}: {e}")
        return None

def implement_rag_system(vector_store, llm):
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "Thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    if vector_store is None or llm is None:
        return None

    try:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return rag_chain
    except Exception as e:
        st.exception(f"Error implementing RAG system: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="Complaint Analysis RAG Chatbot", page_icon="🗣️")

st.header("🗣️ Complaint Analysis RAG Chatbot")
st.markdown("Ask questions about customer complaints and get answers with relevant source documents.")

# Initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

# Initialize the RAG system if not already loaded
if st.session_state.rag_system is None:
    st.session_state.rag_system = initialize_rag_system()

rag_system = st.session_state.rag_system

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Show Sources"):
                st.markdown(message["sources"])

# --- User Input and Response Generation ---
if prompt := st.chat_input("Type your question here..."):
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
                    full_response += f"\n\n*(Response Time: {response_time:.2f} seconds)*"

                    sources_list = []
                    if result.get('source_documents'):
                        for i, doc in enumerate(result['source_documents']):
                            sources_list.append(
                                f"- **Source {i+1} (Complaint ID: {doc.metadata.get('original_complaint_id', 'N/A')}, Product: {doc.metadata.get('product', 'N/A')}):**\n"
                                f"  ```\n{doc.page_content[:500]}...\n  ```" # Display first 500 chars
                            )
                        full_sources = "### Source Documents:\n" + "\n".join(sources_list)
                    else:
                        full_sources = "No relevant source documents found."

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
    # Optionally, re-initialize RAG system if you want to clear its state (unlikely needed)
    # st.session_state.rag_system = None
    # initialize_rag_system() # This would re-run the loading

st.sidebar.button('Clear Chat', on_click=clear_chat_history)

# Provide instructions to run
st.sidebar.markdown(
    """
    ## How to Use:
    1.  Wait for the RAG system to fully load (indicated by success message).
    2.  Type your question in the input box at the bottom.
    3.  Press Enter or click the send button.
    4.  The AI's answer and source documents will appear above.
    5.  Use "Clear Chat" in the sidebar to reset the conversation.
    """
)