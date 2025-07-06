import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# Load cleaned data
df = pd.read_csv('data/filtered_complaints.csv')

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
)

# Choose columns
texts = df['cleaned_narrative'].tolist()
metas = df[['Complaint ID', 'Product']].to_dict(orient='records')

# Split into chunks and maintain metadata
chunks, metadata = [], []
for i, (text, meta) in enumerate(zip(texts, metas)):
    splits = text_splitter.split_text(text)
    chunks.extend(splits)
    metadata.extend([meta] * len(splits))

print(f"Total chunks: {len(chunks)}")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
os.makedirs('vector_store', exist_ok=True)
faiss.write_index(index, 'vector_store/faiss_index.index')

with open('vector_store/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(" Vector index and metadata saved.")
