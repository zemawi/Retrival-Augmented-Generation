import streamlit as st
from src.rag_pipeline import answer_question

st.set_page_config(page_title="CrediTrust AI Assistant", layout="wide")
st.title(" CrediTrust Complaint Assistant")
st.write("Ask a question about customer complaints. The AI will search real complaint data and provide an answer.")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Text input box
question = st.text_input("Your Question", placeholder="e.g., Why are people unhappy with BNPL?", key="input")

# Button row
col1, col2 = st.columns([1, 1])
with col1:
    ask_clicked = st.button("Ask", type="primary")
with col2:
    clear_clicked = st.button("Clear")

# Clear history
if clear_clicked:
    st.session_state.history = []
    st.experimental_rerun()

# Process question
if ask_clicked and question:
    with st.spinner("Searching complaints and generating answer..."):
        result = answer_question(question)
        st.session_state.history.append(result)

# Show results
for i, result in enumerate(reversed(st.session_state.history)):
    st.markdown(f"###  Q{i+1}: {result['question']}")
    st.markdown(f"**Answer:** {result['answer']}")
    with st.expander(" Show Sources"):
        for chunk in result['context'].split("---"):
            st.markdown(f"> {chunk.strip()}")
