from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



#Step1:Load raw PDF(s)

DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
print("Length of PDF pages:",len(documents))
#Step2:Create Chunks
#Step3:Create Vector Embeddings
#Step4:Store embeddings in FAISS