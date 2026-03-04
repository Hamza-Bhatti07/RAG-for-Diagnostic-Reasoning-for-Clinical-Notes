import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. SETUP (Load models only once to save time)
@st.cache_resource
def load_models():
    # Load Retrieval
    index = faiss.read_index("mimic.index")
    with open("mimic_docs.pkl", "rb") as f:
        documents = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load Generator
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    return index, documents, embedder, tokenizer, model

st.title("🏥 Clinical AI Assistant (RAG)")
st.write("Ask a question about the clinical notes database.")

# Load models with a spinner
with st.spinner("Loading AI Brain..."):
    index, documents, embedder, tokenizer, model = load_models()

# 2. UI INPUT
query = st.text_input("Enter your clinical question:")

# 3. RUN RAG
if query:
    # Retrieval
    query_vec = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, k=2)
    
    retrieved_docs = [documents[i] for i in indices[0]]
    context_text = "\n\n".join(retrieved_docs)
    
    # Show retrieved docs (for transparency)
    with st.expander("📄 View Retrieved Patient Notes"):
        st.write(context_text)
    
    # Generation
    prompt = f"Use context to answer. Context: {context_text} Question: {query} Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=2, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display Answer
    st.success(f"**AI Answer:** {answer}")
