import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)

def calculate_cosine_similarity(query_embedding, doc_embedding):
    """
    Calculate cosine similarity between query and document embeddings.
    """
    return float(cosine_similarity([query_embedding], [doc_embedding])[0][0])

@st.cache_resource
def load_model():
    """
    Placeholder for loading the model. Replace with actual model loading if needed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        st.error("SentenceTransformer not installed.")
        return None

# Initialize Streamlit app
st.title("Streamlined Resume Matcher")

# Load the model
model = load_model()
if model is None:
    st.stop()

def process_resumes(uploaded_files):
    """
    Process uploaded JSON files and return resume data.
    """
    combined_data = {}
    for uploaded_file in uploaded_files:
        try:
            data = json.load(uploaded_file)
            if "resumes" in data:
                combined_data.update(data["resumes"])
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            else:
                st.warning(f"No 'resumes' key found in {uploaded_file.name}. Skipping.")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing {uploaded_file.name}: {e}")
    return combined_data

def extract_snippet(resume_text, query_text):
    """
    Extract a snippet from the resume text that matches the query.
    """
    query_words = set(query_text.lower().split())
    sentences = resume_text.split('. ')
    for sentence in sentences:
        if any(word in sentence.lower() for word in query_words):
            return sentence.strip() + '.'
    return "No relevant snippet found."

def compute_match_scores(data, query_text):
    """
    Compute match scores for each resume based on the query.
    """
    query_embedding = model.encode(query_text) if query_text.strip() else None
    matches = []
    if query_embedding is not None:
        for name, resume_data in data.items():
            try:
                doc_embedding = model.encode(resume_data["full_text"])
                score = calculate_cosine_similarity(query_embedding, doc_embedding)
                snippet = extract_snippet(resume_data["full_text"], query_text)
                matches.append((name, score, snippet))
            except Exception as e:
                st.warning(f"Failed to process resume '{name}': {e}")
    return sorted(matches, key=lambda x: x[1], reverse=True)

# Upload and process JSON files
uploaded_files = st.file_uploader("Upload Resume JSON Files (maximum size: 200MB)", type="json", accept_multiple_files=True)
data = {}

if uploaded_files:
    data = process_resumes(uploaded_files)
    st.session_state.data = data

# Search interface
if "data" in st.session_state and st.session_state.data:
    query_text = st.text_area("Enter a query to find matching resumes:", placeholder="e.g., Python developer with SQL experience")

    if st.button("Search"):
        matches = compute_match_scores(st.session_state.data, query_text)

        if matches:
            st.header("Matching Resumes")
            for name, score, snippet in matches:
                st.write(f"### {name}")
                st.write(f"**Match Score:** {score:.2f}")
                st.write(f"**Snippet:** {snippet}")
        else:
            st.warning("No matches found.")
