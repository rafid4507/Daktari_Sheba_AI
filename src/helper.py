from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(data):
    loader = DirectoryLoader(
            data,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata={"source": src}
                )
        )
    return minimal_docs


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    texts_chunks = text_splitter.split_documents(extracted_data)
    return texts_chunks

def download_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embedding = download_embeddings()