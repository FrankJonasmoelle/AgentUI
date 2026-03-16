from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# load all PDFs from the docs/ folder
loader = PyPDFDirectoryLoader("documents/")
documents = loader.load()

# split into chunks (so each chunk fits in context)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# embed and save to chroma_db/
embeddings = OpenAIEmbeddings()
Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

print(f"Ingested {len(chunks)} chunks from {len(documents)} pages")
