import streamlit as st
from heading import get_heading

def about_page():
    get_heading("About This App")

    st.markdown("""
    ## Welcome to the Multi-Page RAG Model App!

    This app allows you to interact with different types of data using Retrieval-Augmented Generation (RAG) models. Below are the main features and a user guide to help you navigate through the app.

    ### Features
    - **PDF/Docx Model**: Upload your PDF or Docx files and ask questions about the data.
    - **Web Model**: Provide a URL and ask questions based on the web page content.
    - **CSV Model**: Upload a CSV file and ask questions about the data.
    - **Session Management**: The app maintains the conversation history for each model, so you can switch between models without losing context. This history is maintained until you reset it.

    ### User Guide
    1. **API Key**: Start by entering your Google API key in the sidebar. If you don't have API key, you can generate it from the given link in `Add Your API Key` popover.
    2. **Select Model**: Use the sidebar to choose from the three RAG models: Document, Web, CSV.
        - **Chat with Documents**: 
          - Upload your PDF or Docx file. Click on the Process button. You can upload multiple files.
          - Select an LLM from the sidebar or you can go with the default LLM.
          - Adjust Temperature and Max tokens limit in the sidebar or leave them to default values.
          - Ask your questions in the input box and get answers based on the data in the files.
        - **Chat with Web**: 
          - Input a webpage URL. Click the Process button.
          - Ask questions about the web page content and get relevant answers.
          - You can change the webpage URL anytime.
        - **Chat with CSV**: 
          - Upload your CSV file.
          - Ask your questions and get responses based on the data in the file.
    3. **Session Management**: 
        - Navigate between different models without losing your conversation history.
        - If you want to reset conversation history of any model, just click the `Reset` button in the sidebar of that model.
    4. **Clear API Key**: 
        - To reset your API key and conversation history, click the "🗑️" button in the sidebar. This will prompt you to enter your API key again.

    ### Connect With Me
    If you have any questions or feedback, please feel free to reach out:
    - [LinkedIn](https://www.linkedin.com/in/noor-ahmad-haral-ml-engineer/)
    - [GitHub](https://github.com/noor-ahmad-haral)
    """)

about_page()