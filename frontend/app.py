"""
Streamlit frontend for the Legal Clause Classifier.

Architecture:
  Streamlit (this file) --HTTP--> FastAPI (src/api/main.py) --> model / Gemini

This file contains NO model-loading or ML logic, and no Gemini calls of its
own. It only sends requests to the API and displays whatever comes back.
Keeping the UI and the ML/LLM logic in separate processes means either can
be changed, restarted, or scaled independently.

Two input modes:
  1. Paste a single clause -> POST /predict, optional POST /explain
  2. Upload a contract (PDF/docx) -> extract text, split into paragraphs
     (see split_into_paragraphs below), classify all via POST /predict-batch,
     show a results table, optional POST /explain on a selected row.

IMPLEMENTATION DECISION FOR THIS PROJECT (not part of the original CUAD
dataset/methodology): CUAD's clause boundaries were drawn by expert legal
annotators reading full contracts. We do not have that. The upload mode
here uses simple paragraph splitting (blank-line-separated blocks) as a
stand-in, purely for demonstration -- far less precise than professional
clause annotation. This is disclosed to the user in the UI, not just here.

Separation of concerns for explanations: the CATEGORY always comes from
our own fine-tuned model (via /predict or /predict-batch). Gemini (via
/explain) only ever explains a category that's already been decided --
it never classifies anything itself.
"""

import os
import re

import requests
import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument

API_URL = os.getenv("API_URL", "http://localhost:8000")
MIN_PARAGRAPH_LENGTH = 20  # filter out headers, page numbers, stray short lines
MAX_PARAGRAPHS = 150  # keep requests reasonably sized
LOW_CONFIDENCE_WARNING_THRESHOLD = 0.5  # median confidence below this -> likely wrong document type


def split_into_paragraphs(text: str) -> list[str]:
    """Split extracted document text into paragraph-like chunks.

    See the module docstring: this is OUR simplified splitting approach
    for this project, not the original CUAD clause segmentation.
    """
    raw_chunks = re.split(r"\n\s*\n", text)
    paragraphs = []
    for chunk in raw_chunks:
        cleaned = re.sub(r"\s+", " ", chunk).strip()
        if len(cleaned) >= MIN_PARAGRAPH_LENGTH:
            paragraphs.append(cleaned)
    return paragraphs[:MAX_PARAGRAPHS]


def call_explain(category: str, text: str) -> str | None:
    try:
        response = requests.post(
            f"{API_URL}/explain", json={"category": category, "text": text}, timeout=30,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the explanation service: {e}")
        return None

    if response.status_code == 200:
        return response.json()["explanation"]
    elif response.status_code == 503:
        st.warning(
            "Explanations aren't available -- the server doesn't have a Gemini API "
            "key configured (see .env.example)."
        )
    else:
        st.error(f"Explanation request failed (status {response.status_code}): {response.text}")
    return None


st.set_page_config(page_title="Legal Clause Classifier", page_icon="⚖️")
st.title("⚖️ AI Legal Contract Clause Classifier")
st.caption(
    "Classify legal contract clauses into categories using a LegalBERT model "
    "fine-tuned on the CUAD (Contract Understanding Atticus Dataset) dataset."
)

with st.expander("ℹ️ How this works (for beginners)"):
    st.markdown(
        """
1. **Text goes in** -- either a clause you paste, or paragraphs extracted from an uploaded document.
2. **Tokenization** -- the text is broken into sub-word pieces and converted into numbers the model understands.
3. **The model** -- a transformer (LegalBERT) fine-tuned on ~12,000 real, labeled contract clauses looks at those numbers.
4. **Prediction** -- the model outputs a probability for each of the 32 clause categories it was trained on; the highest one is the prediction, and that probability is the **confidence score**.
5. **Explanation (optional)** -- a separate AI (Gemini) turns the predicted category into a plain-English description. It does NOT decide the category -- it only explains a category our own model already chose.

**Important scope note:** this model was trained only on commercial contract clauses (licenses, termination, governing law, etc.). It was fine-tuned on isolated, already-extracted clauses, so it works best on clause-length text -- and it was never trained on other legal document types (e.g. court filings, bail applications). Feeding it those will produce unreliable, overconfident-looking results.
        """
    )

mode = st.radio("Input mode", ["Paste a clause", "Upload a contract (PDF/docx)"], horizontal=True)

st.divider()

if mode == "Paste a clause":
    clause_text = st.text_area(
        "Clause text",
        height=180,
        placeholder="e.g. 'This Agreement shall be governed by the laws of the State of Delaware...'",
    )

    if st.button("Predict", type="primary"):
        if not clause_text or len(clause_text.strip()) < 2:
            st.warning("Please enter some clause text first.")
        else:
            with st.spinner("Classifying..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict", json={"text": clause_text}, timeout=30,
                    )
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not reach the API at {API_URL}. Is the FastAPI server running?")
                    response = None
                except requests.exceptions.Timeout:
                    st.error("The API took too long to respond (timed out after 30s).")
                    response = None

            if response is not None:
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["single_result"] = result
                    st.session_state["single_text"] = clause_text
                    st.session_state.pop("single_explanation", None)
                elif response.status_code == 503:
                    st.error(
                        "The model isn't loaded on the server yet. "
                        "Has training finished and was the API restarted since?"
                    )
                else:
                    st.error(f"API returned an error (status {response.status_code}): {response.text}")

    if "single_result" in st.session_state:
        result = st.session_state["single_result"]
        st.success(f"**Predicted category:** {result['category']}")
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        if result["confidence"] < LOW_CONFIDENCE_WARNING_THRESHOLD:
            st.warning(
                "Low confidence -- this text may not resemble any category the model "
                "was trained on. Treat this prediction with caution."
            )

        with st.expander("See all category scores"):
            sorted_scores = sorted(result["all_scores"].items(), key=lambda x: -x[1])
            for category, score in sorted_scores[:10]:
                st.write(f"{category}: {score * 100:.1f}%")

        if st.button("Explain this clause in plain English"):
            with st.spinner("Asking Gemini for a plain-English explanation..."):
                explanation = call_explain(result["category"], st.session_state["single_text"])
            if explanation:
                st.session_state["single_explanation"] = explanation

        if "single_explanation" in st.session_state:
            st.info(f"**Plain-English explanation:**\n\n{st.session_state['single_explanation']}")

else:
    st.info(
        "📌 **Note on this mode:** documents are automatically split into paragraphs "
        "for classification. This is a simplified approach built for this project -- "
        "not the original CUAD methodology (which used expert-annotated clause "
        "boundaries). Results are a demonstration, not a substitute for legal review."
    )

    uploaded_file = st.file_uploader("Upload a contract (PDF or Word .docx)", type=["pdf", "docx"])

    if uploaded_file is not None:
        page_count = None
        with st.spinner("Extracting text..."):
            try:
                if uploaded_file.name.lower().endswith(".pdf"):
                    reader = PdfReader(uploaded_file)
                    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                    page_count = len(reader.pages)
                else:  # .docx
                    doc = DocxDocument(uploaded_file)
                    full_text = "\n\n".join(p.text for p in doc.paragraphs)
            except Exception as e:
                st.error(f"Could not read this file: {e}")
                full_text = ""

        if full_text.strip():
            paragraphs = split_into_paragraphs(full_text)
            source_note = f" (from {page_count} page(s))" if page_count is not None else ""
            st.write(f"Extracted **{len(paragraphs)}** paragraphs to classify{source_note}.")

            with st.expander("Preview extracted paragraphs"):
                for i, p in enumerate(paragraphs[:5]):
                    st.text(f"[{i+1}] {p[:200]}{'...' if len(p) > 200 else ''}")
                if len(paragraphs) > 5:
                    st.caption(f"... and {len(paragraphs) - 5} more")

            if paragraphs and st.button("Classify all paragraphs", type="primary"):
                with st.spinner(f"Classifying {len(paragraphs)} paragraphs..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict-batch", json={"texts": paragraphs}, timeout=120,
                        )
                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not reach the API at {API_URL}. Is the FastAPI server running?")
                        response = None
                    except requests.exceptions.Timeout:
                        st.error("The API took too long to respond (timed out after 120s).")
                        response = None

                if response is not None:
                    if response.status_code == 200:
                        st.session_state["batch_paragraphs"] = paragraphs
                        st.session_state["batch_results"] = response.json()["results"]
                        st.session_state.pop("batch_explanation", None)
                    elif response.status_code == 503:
                        st.error(
                            "The model isn't loaded on the server yet. "
                            "Has training finished and was the API restarted since?"
                        )
                    else:
                        st.error(f"API returned an error (status {response.status_code}): {response.text}")
        else:
            st.warning("No extractable text found in this file (a PDF may be a scanned image without OCR text).")

    if "batch_results" in st.session_state:
        paragraphs = st.session_state["batch_paragraphs"]
        results = st.session_state["batch_results"]
        confidences = sorted(r["confidence"] for r in results)
        median_confidence = confidences[len(confidences) // 2]

        if median_confidence < LOW_CONFIDENCE_WARNING_THRESHOLD:
            st.warning(
                f"⚠️ Median confidence across all paragraphs is only "
                f"{median_confidence * 100:.0f}%. This document may not be a typical "
                "commercial contract (the type this model was trained on) -- treat "
                "these results as unreliable."
            )

        table_rows = [
            {
                "Paragraph (preview)": p[:100] + ("..." if len(p) > 100 else ""),
                "Predicted category": r["category"],
                "Confidence": r["confidence"],
            }
            for p, r in zip(paragraphs, results)
        ]
        table_rows.sort(key=lambda row: row["Confidence"], reverse=True)
        display_rows = [
            {**row, "Confidence": f"{row['Confidence'] * 100:.1f}%"} for row in table_rows
        ]
        st.dataframe(display_rows, width="stretch")

        st.subheader("Explain one paragraph")
        options = [f"[{i+1}] {p[:80]}{'...' if len(p) > 80 else ''}" for i, p in enumerate(paragraphs)]
        selected_idx = st.selectbox(
            "Choose a paragraph to explain in plain English",
            range(len(options)),
            format_func=lambda i: options[i],
        )

        if st.button("Explain selected paragraph"):
            with st.spinner("Asking Gemini for a plain-English explanation..."):
                explanation = call_explain(results[selected_idx]["category"], paragraphs[selected_idx])
            if explanation:
                st.session_state["batch_explanation"] = explanation

        if "batch_explanation" in st.session_state:
            st.info(f"**Plain-English explanation:**\n\n{st.session_state['batch_explanation']}")
