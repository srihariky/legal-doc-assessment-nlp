import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import pandas as pd
import re

# ===================== CONFIG =====================
CLASSIFIER_MODEL = "facebook/bart-large-mnli"
EXPLAINER_MODEL = "google/flan-t5-large" 

RISK_CATEGORIES = [
    "Data Privacy & Tracking",
    "Financial Penalty or Hidden Fees",
    "Arbitration or Waiver of Rights",
    "Unilateral Account Termination",
    "Intellectual Property Surrender"
]

# BATCH SIZE: Number of clauses to process at once.
# If you have >8GB RAM, you can increase this to 16 or 32.
BATCH_SIZE = 8 
# =================================================

st.set_page_config(page_title="The Fine Print Auditor", page_icon="⚖️", layout="wide")

@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else -1
    
    # 1. Load Classifier (The Judge)
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=device)
    
    # 2. Load Explainer (The Translator)
    tokenizer = AutoTokenizer.from_pretrained(EXPLAINER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(EXPLAINER_MODEL)
    if device == 0:
        model = model.to("cuda")
        
    return classifier, tokenizer, model, device

def extract_text(uploaded_file):
    try:
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def segment_text(text):
    # Clean and split into chunks
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.split()) > 8]

def explain_risk_batch(clauses, model, tokenizer, device):
    """
    Generates explanations for a LIST of clauses at once.
    """
    prompts = [
        f"Explain this legal clause in simple English focusing on the risk: {c}" 
        for c in clauses
    ]
    
    inputs = tokenizer(prompts, return_tensors="pt", max_length=512, truncation=True, padding=True)
    if device == 0:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(
        **inputs, 
        max_new_tokens=60, 
        do_sample=False, 
        repetition_penalty=1.5
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def main():
    st.sidebar.title("⚙️ Settings")
    threshold = st.sidebar.slider("Risk Threshold", 0.4, 0.9, 0.60)
    
    st.title("⚡ The Fine Print: High-Speed Auditor")
    st.markdown("**Upload a contract to scan for risks instantly.**")

    # Load models
    with st.spinner("Loading AI Engines..."):
        classifier, tokenizer, explainer_model, device = load_models()

    uploaded_file = st.file_uploader("Upload PDF or TXT", type=['txt', 'pdf'])

    if uploaded_file:
        text = extract_text(uploaded_file)
        if text:
            clauses = segment_text(text)
            st.info(f"Document loaded. Found {len(clauses)} clauses.")
            
            if st.button("🚀 Run Fast Scan"):
                results = []
                progress_bar = st.progress(0)
                status = st.empty()

                # --- BATCH PROCESSING LOOP ---
                total_batches = (len(clauses) + BATCH_SIZE - 1) // BATCH_SIZE
                
                for i in range(0, len(clauses), BATCH_SIZE):
                    batch = clauses[i : i + BATCH_SIZE]
                    
                    # update progress
                    current_batch = (i // BATCH_SIZE) + 1
                    progress_bar.progress(current_batch / total_batches)
                    status.text(f"Processing Batch {current_batch}/{total_batches}...")

                    # 1. Bulk Classify
                    # We send the whole list 'batch' to the classifier at once
                    batch_results = classifier(batch, candidate_labels=RISK_CATEGORIES + ["Neutral"])
                    
                    risky_clauses = []
                    risky_indices = []

                    # 2. Filter for Risks
                    for idx, res in enumerate(batch_results):
                        top_label = res['labels'][0]
                        score = res['scores'][0]
                        
                        if top_label != "Neutral" and score >= threshold:
                            risky_clauses.append(batch[idx])
                            risky_indices.append((top_label, score))

                    # 3. Bulk Explain (Only the risky ones)
                    if risky_clauses:
                        explanations = explain_risk_batch(risky_clauses, explainer_model, tokenizer, device)
                        
                        for j, explanation in enumerate(explanations):
                            label, score = risky_indices[j]
                            results.append({
                                "Risk Type": label,
                                "Confidence": score,
                                "Clause": risky_clauses[j],
                                "Explanation": explanation
                            })

                progress_bar.empty()
                status.empty()

                # --- DISPLAY ---
                if results:
                    st.success(f"Scan Complete! Found {len(results)} risks.")
                    
                    df = pd.DataFrame(results)
                    st.dataframe(df[["Risk Type", "Confidence", "Explanation"]]) # Quick view table
                    
                    for item in results:
                        with st.expander(f"🚩 {item['Risk Type']} ({item['Confidence']:.1%})"):
                            st.write(f"**Explanation:** {item['Explanation']}")
                            st.caption(f"**Original:** {item['Clause']}")
                else:
                    st.balloons()
                    st.success("✅ No risks found.")

if __name__ == "__main__":
    main()