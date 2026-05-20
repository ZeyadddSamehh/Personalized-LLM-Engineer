import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

load_dotenv()

print("--- Loading Engine for Evaluation ---")
DB_DIR = "./ece_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# We use the same Llama 3 model to generate AND evaluate
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True 
)


test_questions = [
    "What is the coefficient of chemical expansion for Li-metal?",
    "What is the primary function of the ETM model?"
]

ground_truths = [
    "The coefficient of chemical expansion for Li-metal is typically around 0.5 to 0.7 depending on temperature.",
    "The ETM model predicts the electro-thermo-mechanical stress behaviors in solid-state batteries to prevent physical cracking."
]

# ==========================================
# 3. GENERATE AI ANSWERS
# ==========================================
print("Generating AI Responses for Test Set")
answers = []
contexts = []

for query in test_questions:
    print("Asking: {query}")
    response = rag_chain.invoke(query)
    
    answers.append(response["result"])
    
    # Ragas needs to know exactly which text chunks the AI read
    source_texts = [doc.page_content for doc in response["source_documents"]]
    contexts.append(source_texts)

# ==========================================
# 4. PREPARE RAGAS DATASET
# ==========================================
data = {
    "question": test_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

# ==========================================
# 5. RUN RAGAS EVALUATION
# ==========================================
print("\n--- Running Mathematical Evaluation (This takes a minute) ---")

# We evaluate based on two critical engineering metrics
result = evaluate(
    dataset = dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
    ],
    llm=llm, # Tell Ragas to use Groq, not OpenAI
    embeddings=embeddings # Tell Ragas to use your HuggingFace embeddings
)

# Convert to a clean Pandas dataframe so you can copy it to your report
df = result.to_pandas()

print("\n" + "="*50)
print("FINAL PERFORMANCE METRICS (Put this in your report!)")
print("="*50)
# Print the average scores
print(f"Overall Faithfulness: {df['faithfulness'].mean():.2f}/1.0")
print(f"Overall Answer Relevancy: {df['answer_relevancy'].mean():.2f}/1.0")
print("="*50)

# Save to a CSV file to easily put in your report
df.to_csv("rag_evaluation_results.csv", index=False)
print("Detailed results saved to 'rag_evaluation_results.csv'")