import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load FAISS index and metadata
index = faiss.read_index("vector_store/faiss_index.index")
with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load the same embedding model as in Task 2
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load a simple LLM for testing (use GPT-2 for local testing)
llm = pipeline("text-generation", model="gpt2", max_new_tokens=150)

# Prompt template function
def build_prompt(context, question):
    return f"""You are a financial analyst assistant for CrediTrust.
Use the following retrieved complaint excerpts to answer the question. If the context doesn't help, say so.

Context:
{context}

Question: {question}
Answer:"""

# Main RAG function
def answer_question(question, k=5):
    # Step 1: Embed the question
    question_vector = embed_model.encode([question])
    
    # Step 2: Search vector index
    distances, indices = index.search(np.array(question_vector), k)

    # Step 3: Extract top-k chunks from metadata
    chunks = [metadata[i] for i in indices[0]]
    context = "\n---\n".join([f"{i+1}. {chunk['Product']} (ID: {chunk['Complaint ID']})" for i, chunk in enumerate(chunks)])
    
    # Step 4: Build prompt
    prompt = build_prompt(context, question)
    
    # Step 5: Generate response from LLM
    response = llm(prompt)[0]['generated_text']
    return {
        "question": question,
        "answer": response.split("Answer:")[-1].strip(),
        "context": context
    }

# Run this if script is executed directly
if __name__ == "__main__":
    q = "Why are customers unhappy with BNPL?"
    result = answer_question(q)
    print(" Question:", result["question"])
    print(" Answer:", result["answer"])
    print("\n Retrieved Context:\n", result["context"])
