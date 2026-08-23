import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(
    page_title="Vespera DNS Tunneling IDS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit Chrome padding & elements for clean full-screen HTML integration
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        iframe {
            border: none;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Load index.html from project directory
html_path = os.path.join(os.path.dirname(__file__), "index.html")

if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=1400, scrolling=True)
else:
    st.error("index.html file not found in repository.")
