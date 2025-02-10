from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Step1:Load raw PDF(s)

DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents


#print("Loading PDF files...")
documents = load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages:",len(documents))

#Step2:Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50) 
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
#print("Splitting text into chunks...")
text_chunks = create_chunks(extracted_data=documents)
#print("length of text chunks:",len(text_chunks))


#Step3:Create Vector Embeddings

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
#print("Loading embedding model...")
embedding_model = get_embedding_model()
#print("Embedding model loaded successfully.")
# Step 4: Store embeddings in FAISS (Batch-wise Processing)
DB_FAISS_PATH = "vectorstore/db_faiss"
batch_size = 1000  # Process in batches
db = None  # Initialize FAISS

#print(f"Processing in batches of {batch_size}...")

for i in range(0, len(text_chunks), batch_size):
    batch = text_chunks[i:i+batch_size]
    print(f"Processing batch {i}-{i+batch_size}...")

    if db is None:
        db = FAISS.from_documents(batch, embedding_model)  # First batch initializes FAISS
    else:
        db.add_documents(batch)  # Add new batch to FAISS index

#print("Saving FAISS database...")
db.save_local(DB_FAISS_PATH)
#print("FAISS database saved successfully!")