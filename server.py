import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()

# 1. INITIALIZE RETRIEVAL SYSTEM 

print("--- Connecting to Existing Database ---")
DB_DIR = "./ece_index"

# Ensure the DB actually exists before starting
if not os.path.exists(DB_DIR):
    raise RuntimeError(f"Database not found at {DB_DIR}. Please run ingest.py first!")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    temperature=0, 
    max_tokens=1024
)

# We will pass the history directly from the UI or Terminal loop.
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True 
)

# 2. SETUP FASTAPI 

app = FastAPI(title="Engineering RAG API")

# We define exactly what the UI needs to send us
class ChatRequest(BaseModel):
    query: str
    # UI sends history as a list of [User Question, AI Answer] pairs
    chat_history: List[Tuple[str, str]] = [] 

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Pass BOTH the new question and the past history from the UI to the AI
    response = rag_chain.invoke({
        "question": request.query,
        "chat_history": request.chat_history
    })
    
    # Page Referencing Logic
    sources = response.get("source_documents", [])
    unique_refs = set()
    formatted_refs = []
    
    for doc in sources:
        book_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page_num = doc.metadata.get('page', 0) + 1 # PDFs are 0-indexed
        ref = f"{book_name} | Page: {page_num}"
        
        if ref not in unique_refs:
            formatted_refs.append(ref)
            unique_refs.add(ref)
            
    # Send the answer and references back to the UI
    return {
        "answer": response["answer"],
        "references": formatted_refs
    }


# 3. EXECUTION BLOCK (The "Ignition")

def run_local_terminal_chat():
    print("\n" + "="*40)
    print("LOCAL TERMINAL CHAT MODE STARTED")
    print("Type 'exit' or 'quit' to stop.")
    print("="*40)
    
    local_history = []
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        # Call the chain with the query and the local history
        response = rag_chain.invoke({
            "question": user_query, 
            "chat_history": local_history
        })
        
        print("\nAI:", response["answer"])
        
        # --- THE MISSING METADATA LOGIC ---
        print("\n--- TECHNICAL REFERENCES ---")
        sources = response.get("source_documents", [])
        if not sources: 
            print("No documents retrieved. Please run the Ingesting file or check refrences in file")
        else:
            unique_refs = set()
            for doc in sources:
                book_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = doc.metadata.get('page', 0) + 1 # PDFs are 0-indexed
                doc_type = doc.metadata.get('doc_type', 'unknown')
                
                ref = f"[{doc_type.upper()}] Source: {book_name} | Page: {page_num}"
                if ref not in unique_refs:
                    print(f"-> {ref}")
                    unique_refs.add(ref)
        print("========================================")
        # ----------------------------------
        
        # Add the conversation to history so the AI remembers it next time
        local_history.append((user_query, response["answer"]))

if __name__ == "__main__":
    import sys
    
    # Check if we asked for the terminal version
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_local_terminal_chat()
    else:
        # Otherwise, start the FastAPI server for the UI
        import uvicorn
        print("--- Starting FastAPI Server on Port 8000 ---")
        uvicorn.run(app, host="0.0.0.0", port=8000)