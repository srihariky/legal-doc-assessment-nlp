import os
import re
import csv
import torch
from tqdm import tqdm  # <--- NEW: Progress bar
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader 

# ===================== SYSTEM CONFIG =====================
# The Judge: Zero-Shot Classifier
CLASSIFIER_MODEL = "facebook/bart-large-mnli" 

# The Brain Upgrade: Much smarter, slightly slower
EXPLAINER_MODEL = "google/flan-t5-large"

RISK_CATEGORIES = [
    "Data Privacy & Tracking", 
    "Financial Penalty or Hidden Fees", 
    "Arbitration or Waiver of Rights", 
    "Unilateral Account Termination",
    "Intellectual Property Surrender"
]
RISK_THRESHOLD = 0.60  # Adjusted slightly for the smarter model
# =========================================================

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def load_document(file_path):
    # Remove quotes if user dragged file into terminal
    file_path = file_path.strip().strip('"')
    
    if file_path.lower().endswith('.pdf'):
        print(f"[Input] Detected PDF. Extracting text layers...")
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        print(f"[Input] Detected text file. Reading directly...")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print("❌ Error: Unsupported file format. Use .txt or .pdf")
        return None

def segment_into_clauses(text):
    text = re.sub(r'\s+', ' ', text).strip()
    # Split by sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clauses = []
    for sent in sentences:
        if len(sent.split()) > 8: # Filter out page numbers/headers
            clauses.append(sent.strip())
    return clauses

def explain_clause_manual(clause, model, tokenizer, device):
    """
    Uses the Large model to generate a high-quality summary.
    """
    # PROMPT ENGINEERING: 
    # We give the larger model a more complex instruction.
    prompt = (
        f"Explain the following legal clause in plain English. "
        f"Focus on what the user is losing or risking.\n\n"
        f"Legal Clause: {clause}\n\n"
        f"Plain English Explanation:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    if device == 0:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # GENERATION PARAMETERS:
    # Tuned for "T5-Large" to prevent repetition and encourage clarity
    outputs = model.generate(
        **inputs, 
        max_new_tokens=80, 
        do_sample=True,          # Allow some creativity
        temperature=0.7,         # Not too random, not too robotic
        repetition_penalty=1.5,  # Force it to use new words
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("="*60)
    print(" 🧠  THE FINE PRINT: Legal Risk Assessor (Brain Upgrade)")
    print("="*60)
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"\n[Hardware] Running on: {'GPU 🚀' if device == 0 else 'CPU (This might be slow)'}")

    file_path = input("\nEnter path to file (.txt or .pdf): ").strip()
    if not os.path.exists(file_path.strip('"')):
        print("❌ Error: File not found.")
        return
    
    text = load_document(file_path)
    if not text or len(text) < 50:
        print("❌ Error: Could not extract text.")
        return

    print("\n[Booting AI Engines] Loading T5-Large (Wait for it)...")
    
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=device)
    
    tokenizer = AutoTokenizer.from_pretrained(EXPLAINER_MODEL)
    explainer_model = AutoModelForSeq2SeqLM.from_pretrained(EXPLAINER_MODEL)
    if device == 0:
        explainer_model = explainer_model.to("cuda")

    clauses = segment_into_clauses(text)
    print(f"\n[Analysis] Extracted {len(clauses)} clauses.")
    print("[Analysis] Starting Deep Scan...\n")

    report = []

    # PROGRESS BAR LOOP
    # We use 'tqdm' to show a progress bar
    for clause in tqdm(clauses, desc="Analyzing Clauses", unit="clause"):
        
        # 1. Classify Risk
        result = classifier(clause, candidate_labels=RISK_CATEGORIES + ["Neutral"])
        top_label = result['labels'][0]
        score = result['scores'][0]

        # 2. Explain if Risky
        if top_label != "Neutral" and score >= RISK_THRESHOLD:
            simple_english = explain_clause_manual(clause, explainer_model, tokenizer, device)
            
            report.append({
                "Risk Category": top_label,
                "Confidence": f"{score:.1%}",
                "Simple Explanation": simple_english,
                "Original Text": clause
            })

    # FINAL OUTPUT
    print("\n\n" + "="*60)
    print(f"🚨 SCAN COMPLETE: Found {len(report)} Potential Risks")
    print("="*60 + "\n")

    if report:
        for idx, item in enumerate(report, 1):
            print(f"[{idx}] 🚩 {item['Risk Category'].upper()}")
            print(f"    🗣️ {item['Simple Explanation']}")
            print(f"    📜 \"{item['Original Text'][:100]}...\"\n")

        with open("risk_report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=report[0].keys())
            writer.writeheader()
            writer.writerows(report)
        print("✅ Report saved to 'risk_report.csv'")
    else:
        print("✅ No risks found. (Or the document was too safe!)")

if __name__ == "__main__":
    main()