import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from heading import get_heading
from langchain_community.vectorstores import FAISS

get_heading("Chat With Web")



gemini_api_key = st.session_state.google_api_key


def reset_webpage_chat():
    keys_to_reset = [
        "chat_history_web",
        "file_processed",
        "vector_store",
        "current_url"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.chat_history_web = [
        AIMessage(content="Hi there! How can I help you today?"),
    ]


def get_vector_store_from_url(url):
    # load the content from the website
    loader = WebBaseLoader(url)
    documents = loader.load()
    # split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
         chunk_size= 500,
        chunk_overlap= 100,
        length_function=len)
    
    doc_chunks = text_splitter.split_documents(documents)

    # create a vector store
    vector_store = FAISS.from_documents(doc_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    return  vector_store


def get_context_retriever_chain(vector_store):
    # llm model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key= gemini_api_key)
    # create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})

    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history_web"),
        ("human", "{input}")
        
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key= gemini_api_key)

    prompt = """You are a helpful assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. Do not mention
    any source for the documents.
    If you don't know the answer, just say that you don't know. \n\n

    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder(variable_name="chat_history_web"),
        ("user", "{input}")
    ])
    
    stuff_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_docs_chain)
 

def get_response(user_query):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
            "chat_history_web": st.session_state.chat_history_web,
            "input": user_query
        })

    # st.sidebar.write(response['context'])
    return response['answer']



if "chat_history_web" not in st.session_state:
        st.session_state.chat_history_web = [
            AIMessage(content="Hi there! How can I help you today?"),
        ]

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "current_url" not in st.session_state:
    st.session_state.current_url = ""


with st.sidebar:
    website_url = st.text_area("Add a Webpage URL", value=st.session_state.current_url)

    columns =st.columns(2, vertical_alignment="center")
    with columns[0]:
        process_button = st.button("Process", use_container_width=True, type="primary")

    if process_button and website_url != st.session_state.current_url:
        st.session_state.current_url = website_url

        with st.spinner("Processing..."):
            try:
                st.session_state.vector_store = get_vector_store_from_url(website_url)
                st.session_state.file_processed = True
            except Exception as e:
                st.write(f"An Error has occurred \n\n{e}")

    with columns[1]:
        st.button(
        "🗑️ Reset", 
        on_click=reset_webpage_chat,
        use_container_width=True
        )       


for message in st.session_state.chat_history_web:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="bot.png"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="human.png"):
            st.write(message.content)


if st.session_state.file_processed:

    if user_query := st.chat_input("Ask me questions..."):
        st.session_state.chat_history_web.append(HumanMessage(content=user_query))

        with st.chat_message("Human", avatar="human.png"):
            st.write(user_query)

        with st.spinner("Thinking..."):
            try:
                response = get_response(user_query)
    
                with st.chat_message("AI", avatar="bot.png"):
                    st.write(response)
                    
            except Exception as e:
                 st.write(f"An Error has occurred \n\n{e}")

        st.session_state.chat_history_web.append(AIMessage(content=response))

else:
    st.info("Please provide a webpage URL to start chatting.")
