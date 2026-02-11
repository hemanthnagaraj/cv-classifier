#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import tempfile
import os
import pdfplumber  # For PDF text extraction

# Load model
@st.cache_resource
def load_model():
    return joblib.load('cv_classifier_nb.pkl')

model_data = load_model()
pipeline = model_data['pipeline']

# To extract text from files
def extract_text_from_file(uploaded_file):
    "Extract text from .txt, .docx, .doc, .pdf files"
    text = ""
    
    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'txt':
            text = uploaded_file.read().decode('utf-8')
        
        elif file_extension == 'docx':
            from docx import Document
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            doc = Document(tmp_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            os.unlink(tmp_path)
        
        elif file_extension == 'doc':
            from docx import Document
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            doc = Document(tmp_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            os.unlink(tmp_path)
        
        elif file_extension == 'pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            with pdfplumber.open(tmp_path) as pdf:
                text = '\n'.join([page.extract_text() for page in pdf.pages if page.extract_text()])
            
            os.unlink(tmp_path)
        
        else:
            text = f"Unsupported file format: .{file_extension}"
            
    except Exception as e:
        text = f"Error reading file: {str(e)}"
    
    return text

# App UI
st.title("CV Classifier")
st.markdown("Upload a CV file or paste text to classify job category")

# File Upload
st.subheader("Upload CV ")
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['txt', 'docx', 'doc', 'pdf'],
    help="Upload .txt, .docx, .doc or .pdf files"
)

cv_text = ""

if uploaded_file:
    with st.spinner(f"Extractinf the text from {uploaded_file.name}..."):
        cv_text = extract_text_from_file(uploaded_file)
    
    if "Error" not in cv_text and "Unsupported" not in cv_text:
        st.success(f"File loaded: {uploaded_file.name}")
        
        # Show file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Type", f".{uploaded_file.name.split('.')[-1]}")
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("Text Length", f"{len(cv_text):,} chars")
        
        # Show extracted text
        with st.expander("📋 View extracted text"):
            st.text(cv_text[:1000] + "..." if len(cv_text) > 1000 else cv_text)
    else:
        st.error(cv_text)

# Manual Text Input (in case document not available in the accepted format)
st.subheader("Or Paste CV Text")
manual_text = st.text_area("Paste text here:", height=150)

# Use whichever is available
if not cv_text and manual_text:
    cv_text = manual_text

# Predict when text is provided
if cv_text and "Error" not in cv_text and "Unsupported" not in cv_text:
    # Predict
    prediction = pipeline.predict([cv_text])[0]
    probabilities = pipeline.predict_proba([cv_text])[0]
    
    # Display results
    st.success(f"**Predicted Category:** {prediction}")
    
    # Show confidence
    confidence = max(probabilities)
    st.metric("Confidence", f"{confidence:.1%}")
    
    # Show all probabilities
    st.subheader("All Category Probabilities:")
    for label, prob in zip(pipeline.classes_, probabilities):
        col1, col2, col3 = st.columns([2, 5, 2])
        with col1:
            st.write(f"**{label}**")
        with col2:
            st.progress(float(prob))
        with col3:
            st.write(f"{prob:.1%}")


st.markdown("---")
st.caption("Model: Naive Bayes CV classifier")