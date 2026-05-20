import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths
BOOKS_DIR = r"D:\Zeyad Folder\Zeyad Uni\Years\Year 4\Term 8\Introduction to Artificial Intelligence\Project\Personalized-LLM-Engineer\Books"
RESEARCH_DIR = r"D:\Zeyad Folder\Zeyad Uni\Years\Year 4\Term 8\Introduction to Artificial Intelligence\Project\Personalized-LLM-Engineer\Research"
DB_DIR = "./ece_index"

def ingest_data():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    # 1. Ask the database what files it already has
    print("Checking existing database records...")
    existing_data = vector_db.get(include=["metadatas"])
    
    # Create a fast lookup set of all existing file paths
    existing_files = set()
    for meta in existing_data["metadatas"]:
        # Chroma sometimes returns None for empty metadata, so we check first
        if meta is not None and "source" in meta:
            existing_files.add(meta["source"])
            
    print(f"Found {len(existing_files)} unique files already ingested.")
    # Define our sources
    sources = [
        {"path": BOOKS_DIR, "type": "foundation"},
        {"path": RESEARCH_DIR, "type": "research"}
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    BATCH_Max = 5000 

    for source in sources:
        pdf_files = glob.glob(os.path.join(source["path"], "*.pdf"))
        for pdf in pdf_files:
            print(f"Ingesting [{source['type']}]: {os.path.basename(pdf)}")
            
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            
            # Add metadata tags before splitting
            for doc in docs:
                doc.metadata["doc_type"] = source["type"]
                doc.metadata["importance"] = "high" if source["type"] == "research" else "normal"

            chunks = text_splitter.split_documents(docs)
            total_chunks = len(chunks)
            
            for i in range(0, total_chunks, BATCH_Max):
                # Slice the LangChain Document objects
                batch = chunks[i : i + BATCH_Max]
                
                # Insert just this batch
                vector_db.add_documents(batch)
                
                current_batch = (i // BATCH_Max) + 1
                total_batches = (total_chunks + BATCH_Max - 1) // BATCH_Max
                print(f"  Inserted batch {current_batch}/{total_batches}")
            
    print("Ingestion finished")

if __name__ == "__main__":
    ingest_data()