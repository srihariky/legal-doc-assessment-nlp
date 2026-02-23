import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import pandas as pd
import re

# ===================== CONFIG =====================
# Swapped to 'distilbart' (Much smaller, still very smart)
CLASSIFIER_MODEL = "valhalla/distilbart-mnli-12-3" 

# Swapped back to 'base' so it fits in the 1GB cloud RAM limit
EXPLAINER_MODEL = "google/flan-t5-base"  

RISK_CATEGORIES = [
    "Data Privacy & Tracking",
    "Financial Penalty or Hidden Fees",
    "Arbitration or Waiver of Rights",
    "Unilateral Account Termination",
    "Intellectual Property Surrender"
]
# =================================================

# --- PAGE SETUP ---
st.set_page_config(
    page_title="The Fine Print Auditor",
    page_icon="⚖️",
    layout="wide"
)

# --- CACHED MODEL LOADING (CRITICAL) ---
# We use @st.cache_resource so models load ONCE and stay in memory.
# Without this, the app would reload the 2GB models on every click!
@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else -1
    
    # 1. Load Classifier
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=device)
    
    # 2. Load Explainer
    tokenizer = AutoTokenizer.from_pretrained(EXPLAINER_MODEL)
    explainer_model = AutoModelForSeq2SeqLM.from_pretrained(EXPLAINER_MODEL)
    if device == 0:
        explainer_model = explainer_model.to("cuda")
        
    return classifier, tokenizer, explainer_model, device

# --- HELPER FUNCTIONS ---
def extract_text(uploaded_file):
    """Handles both PDF and TXT file uploads."""
    try:
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            # Assume text file
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def segment_text(text):
    """Clean and split text into clauses."""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.split()) > 8]

def explain_risk(clause, model, tokenizer, device):
    prompt = (
        f"Explain the following legal clause in plain English. "
        f"Focus on the risk to the user.\n\n"
        f"Legal Clause: {clause}\n\n"
        f"Plain English Explanation:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    if device == 0:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(
        **inputs, max_new_tokens=80, do_sample=True, 
        temperature=0.7, repetition_penalty=1.5, early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- MAIN APP UI ---
def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.info("Models loaded & ready.")
    threshold = st.sidebar.slider("Risk Sensitivity", 0.4, 0.9, 0.60)
    
    # Main Content
    st.title("⚖️ The Fine Print: AI Legal Auditor")
    st.markdown("""
    **Don't sign what you don't understand.** Upload a Contract, Terms of Service, or Privacy Policy to detect potential traps.
    """)

    # 1. Load Models (Cached)
    with st.spinner("Waking up the AI... (First run takes 30s)"):
        classifier, tokenizer, explainer_model, device = load_models()

    # 2. File Uploader
    uploaded_file = st.file_uploader("Drop your document here (PDF or TXT)", type=['txt', 'pdf'])

    if uploaded_file:
        # Extract Text
        text = extract_text(uploaded_file)
        
        if text:
            st.success(f"File loaded! Found {len(text)} characters.")
            
            if st.button("🔍 Scan for Risks"):
                clauses = segment_text(text)
                results = []
                
                # Progress Bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, clause in enumerate(clauses):
                    # Update progress
                    progress = (i + 1) / len(clauses)
                    progress_bar.progress(progress)
                    status_text.text(f"Scanning clause {i+1}/{len(clauses)}...")
                    
                    # AI Analysis
                    output = classifier(clause, candidate_labels=RISK_CATEGORIES + ["Neutral"])
                    top_label = output['labels'][0]
                    score = output['scores'][0]
                    
                    if top_label != "Neutral" and score >= threshold:
                        explanation = explain_risk(clause, explainer_model, tokenizer, device)
                        results.append({
                            "Risk Type": top_label,
                            "Confidence": score,
                            "Clause": clause,
                            "Explanation": explanation
                        })
                
                status_text.text("Scan Complete!")
                progress_bar.empty()
                
                # --- RESULTS DISPLAY ---
                if results:
                    st.divider()
                    st.subheader(f"🚨 Detected {len(results)} Potential Risks")
                    
                    # Convert to DataFrame for CSV download
                    df = pd.DataFrame(results)
                    csv = df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        "📥 Download Full Report (CSV)", 
                        data=csv, 
                        file_name="legal_risk_report.csv", 
                        mime="text/csv"
                    )
                    
                    # Display Cards for each risk
                    for item in results:
                        with st.expander(f"🚩 {item['Risk Type'].upper()} ({item['Confidence']:.0%})", expanded=True):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("**📜 Original Text**")
                                st.caption(item['Clause'])
                                
                            with col2:
                                st.markdown("**🗣️ Simple English**")
                                st.error(item['Explanation'])
                else:
                    st.balloons()
                    st.success("✅ No significant risks found! This document looks safe.")

if __name__ == "__main__":
    main()