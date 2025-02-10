import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def load_llm():
    """Load the HuggingFace LLM using InferenceClient"""
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    return InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

def query_llm(question, context, llm):
    prompt = f"""
    Use the information provided in the context to answer the user's question.
    If you don't know the answer, say "I don't know."
    
    Context: {context}
    Question: {question}
    
    Start the answer directly.
    """
    response = llm.text_generation(prompt, max_new_tokens=200)
    return response

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        llm = load_llm()

        try:
            vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            retrieved_docs = retriever.invoke(prompt)
            retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

            result = query_llm(prompt, retrieved_context, llm)

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
