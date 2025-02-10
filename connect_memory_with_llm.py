import os
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Setup HuggingFace API & Model
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm():
    """Load the HuggingFace LLM using InferenceClient"""
    return InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

client = load_llm()

# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def query_llm(question, context):
    prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context, question=question)
    response = client.text_generation(prompt, max_new_tokens=200)
    return response

# Step 3: Load FAISS Database
DB_FAISS_PATH = os.path.abspath("vectorstore/db_faiss")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading FAISS database: {e}")
    exit(1)

retriever = db.as_retriever(search_kwargs={'k': 3})

# Step 4: Retrieve Context & Query LLM
user_query = input("Write Query Here: ")

# Retrieve relevant context from FAISS
retrieved_docs = retriever.invoke(user_query)
retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

try:
    response = query_llm(user_query, retrieved_context)
    print("\nRESULT:", response)
    #print("\nSOURCE DOCUMENTS:", retrieved_docs)
except Exception as e:
    print(f"Error during query execution: {e}")
