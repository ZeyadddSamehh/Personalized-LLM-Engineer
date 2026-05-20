# Intro-to-AI-project
Build a Technical Retrieval-Augmented Generation (RAG) System that uses engineering research papers and references to answer queries. 

Project Scope:
    -Deep learning
    -Transformers
    -AI-agents and tool use

Project Requirments:
1) Real-World problem Focus:
        -LLMs often do a poor job at answering complex engineering questions due to either the lack of researhc papers or refrences.
        -Engineers require a tool tuned specifically to them that has in it's database a plethora of well made refrences to aid the engineer
        -an LLM tuned to the needs of an engineer would tremendously cut thier efforts in a big way in retreiving data from refrences and help them understand concepts deeply
        -Although LLMs are very useful, they cannot be relied on entirely
        -This tool is grounded to answer only questions related to engineering efficiently and with no fluff.
        -Furthermore, large files and sensitive information might be at risk when using public LLMs such as OpenAIs chatgpt or Google's Gemini.

2) Model Design and Justification
        -We chose a Retrieval-Augmented Generation (RAG) architecture instead of fine-tuning a model, as RAG anchors the AI strictly to verified technical literature, greatly reducing hallucinations without         requiring expensive model retraining.
        -The system architecture consists of a 3-tier stack: a Streamlit UI frontend, a persistent FastAPI server backend, and a ChromaDB vector database storing chunked PDFs.
        -Our core inference model is Llama-3.3-70b accessed via the Groq API to provide the near-instant inference speeds necessary for real-time engineering chat.
        -We implemented a Two-Stage Retrieval algorithm to maximize precision. Stage 1 uses a fast HuggingFace bi-encoder (all-MiniLM-L6-v2) to fetch 15 candidate chunks via cosine similarity. Stage 2 uses a             cross-encoder reranker (ms-marco-MiniLM-L-6-v2) to logically score and compress those candidates down to the absolute best 4.
        -Key hyperparameters: We utilized a chunk_size of 1000 characters (approx. one engineering paragraph) to prevent slicing equations or definitions in half, and a chunk_overlap of 200 characters to ensure             context isn't lost across page breaks.
        -Alternative considered: Fine-tuning a local LLM was considered, but rejected because it remains prone to hallucination and cannot easily point to specific page numbers in textbooks, which is a hard             requirement for verifying engineering formulas.
        -For explainability and interpretability, the system relies on physical source mapping. Every generated answer automatically extracts document metadata to cite the exact book name and page number, proving exactly where the model's logic originated.

3) Implementation
        -The project heavily utilizes industry-standard libraries: LangChain (for RAG orchestration), HuggingFace (for local embeddings and reranking), Chroma (for persistent vector storage), and         FastAPI/Streamlit (for network routing and user interface).
        -The codebase is original and structurally divided into an offline ingestion pipeline and a live retrieval server to manage RAM efficiently when parsing large 800-page textbooks via lazy loading.
        -To ensure full reproducibility for the instructor, the Python environment is locked and documented using a comprehensive requirements.txt file containing the exact versions of all dependencies.
        -Prompt engineering is natively handled via LangChain's ConversationalRetrievalChain, which automatically evaluates the user's current question against the UI's chat history to generate standalone, context-aware queries before executing the vector search.
