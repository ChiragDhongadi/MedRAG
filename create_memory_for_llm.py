from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load all PDFs from a directory
loader = DirectoryLoader(
    path="data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

text_chunks = text_splitter.split_documents(docs)

# Create embeddings for the text chunks
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the text chunks and save it locally
BD_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(BD_FAISS_PATH)

