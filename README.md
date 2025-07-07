# Intelligent Complaint Analysis for Financial Services
# Powered by Retrieval-Augmented Generation (RAG)
**A project for CrediTrust Financial to turn customer complaint data into real-time, evidence-backed insights using AI, embeddings, and semantic search.**
# Overview
This project builds an internal AI chatbot that enables product managers, support, and compliance teams to understand real customer pain points across five financial product categories:

* Credit Cards

* Personal Loans

* Buy Now, Pay Later (BNPL)

* Savings Accounts

* Money Transfers
# The system uses:
* Real-world complaint data (from the CFPB)
* Text chunking and cleaning
* Sentence-transformer embeddings
*  FAISS vector search
*  RAG pipeline with LLM for natural-language querying
*  Streamlit/Gradio app for user interface

# Tech Stack
* Python 3.13
* pandas
* sentence-transformers
* faiss-cpu
* LangChain
* Streamlit or Gradio
* Hugging Face Transformers

  # Running the Project
1. Install dependencies:
   pip install -r requirements.txt
2. Run EDA/preprocessing:
   jupyter notebook notebooks/eda_preprocessing.ipynb
3. Generate embeddings:
   python src/embedding_pipeline.py
4. Launch chatbot interface:
   streamlit run app.py

