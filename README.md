# 🚀 Multi-Page RAG Model App

Welcome to the Multi-Page RAG Model App! This app allows you to interact with different types of data using Retrieval-Augmented Generation (RAG) models. Below you'll find an overview of the app, its features, a user guide, and instructions for setting it up locally.

## 🌟 Features

- **📄 PDF/Docx Model**: Upload your PDF or Docx files and ask questions about the data.
- **🌐 Web Model**: Provide a URL and ask questions based on the web page content.
- **📊 CSV Model**: Upload a CSV file and ask questions about the data.
- **🔄 Session Management**: The app maintains the conversation history for each model, so you can switch between models without losing context. This history is maintained until you reset it.

## 📋 User Guide

### 🔑 API Key
1. Start by entering your Google API key in the sidebar. If you don't have an API key, you can generate it from the provided link in the `Add Your API Key` popover.

### 🔀 Select Model
2. Use the sidebar to choose from the three RAG models: Document, Web, CSV.

#### 📄 Chat with Documents
- Upload your PDF or Docx files. Click on the Process button. You can upload multiple files.
- Select an LLM from the sidebar or use the default LLM.
- Adjust Temperature and Max tokens limit in the sidebar or leave them to default values.
- Ask your questions in the input box and get answers based on the data in the files.

#### 🌐 Chat with Web
- Input a webpage URL. Click the Process button.
- Ask questions about the web page content and get relevant answers.
- You can change the webpage URL anytime.

#### 📊 Chat with CSV
- Upload your CSV file.
- Ask your questions and get responses based on the data in the file.

### 🔄 Session Management
3. Navigate between different models without losing your conversation history.
4. If you want to reset the conversation history of any model, just click the `🗑️ Reset` button in the sidebar of that model.

### ❌ Clear API Key
5. To reset your API key and conversation history, click the `🗑️` button in the sidebar. This will prompt you to enter your API key again.

### 📞 Connect With Me
If you have any questions or feedback, please feel free to reach out:
- [LinkedIn](https://www.linkedin.com/in/noor-ahmad-haral-ml-engineer/)
- [GitHub](https://github.com/noor-ahmad-haral)

## ⚙️ Installation

To run this app locally, follow these steps:

1. **Clone the repository:**
   git clone https://github.com/noor-ahmad-haral/multi-page-rag-model-app.git
   cd multi-page-rag-model-app

2. **Create the environment:**
   python -m venv venv

3. **Activate the virtual environment:**

   On Windows:
   .\venv\Scripts\activate

   On macOS/Linux:
   source venv/bin/activate

4. **Install the required packages:**
   pip install -r requirements.txt

5. **Run the Streamlit app:**
   streamlit run app.py
