import streamlit as st

def get_heading(text):
    text = f"""
<h1 style='text-align: center;
            background-image: linear-gradient(to right, #347deb, #ff4747);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: Montserrat, sans-serif;
            '>{text}</h1>
"""

    st.markdown(text, unsafe_allow_html=True)
