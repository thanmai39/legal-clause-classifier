"""
Streamlit frontend for the Legal Clause Classifier.

Architecture:
  Streamlit (this file) --HTTP--> FastAPI (src/api/main.py) --> model

This file contains NO model-loading or ML logic. It only sends text to the
API and displays whatever comes back. Keeping the UI and the ML logic in
separate processes means either can be changed, restarted, or scaled
independently -- e.g. the API could later run on a different, more
powerful machine than the UI.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Legal Clause Classifier", page_icon="⚖️")
st.title("⚖️ AI Legal Contract Clause Classifier")
st.caption(
    "Paste a contract clause below to classify it into a legal category. "
    "Built on a LegalBERT model fine-tuned on the CUAD dataset."
)

clause_text = st.text_area(
    "Clause text",
    height=180,
    placeholder="e.g. 'This Agreement shall be governed by the laws of the State of Delaware...'",
)

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if not clause_text or len(clause_text.strip()) < 2:
        st.warning("Please enter some clause text first.")
    else:
        with st.spinner("Classifying..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": clause_text},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the API at {API_URL}. "
                    "Is the FastAPI server running?"
                )
                response = None
            except requests.exceptions.Timeout:
                st.error("The API took too long to respond (timed out after 30s).")
                response = None

        if response is not None:
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Predicted category:** {result['category']}")
                st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

                with st.expander("See all category scores"):
                    sorted_scores = sorted(
                        result["all_scores"].items(), key=lambda x: -x[1]
                    )
                    for category, score in sorted_scores[:10]:
                        st.write(f"{category}: {score * 100:.1f}%")

            elif response.status_code == 503:
                st.error(
                    "The model isn't loaded on the server yet. "
                    "Has training finished and was the API restarted since?"
                )
            else:
                st.error(f"API returned an error (status {response.status_code}): {response.text}")
