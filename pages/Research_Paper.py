import streamlit as st
import base64
import os

st.set_page_config(page_title="Research Paper", layout="wide", initial_sidebar_state="collapsed")

# Read the PDF file
pdf_path = os.path.join(os.path.dirname(__file__), "..", "public", "research paper.pdf")
if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # Embed the PDF in full screen
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#toolbar=1&navpanes=0&scrollbar=1" width="100%" height="1000px" style="border: none;"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
else:
    st.error("Research paper not found.")
