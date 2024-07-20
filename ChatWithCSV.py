import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_core.messages import AIMessage, HumanMessage
import io
from heading import get_heading


get_heading("Chat With CSV")

gemini_api_key = st.session_state.google_api_key
# gemini_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1,
                                max_output_tokens=1000,
                                google_api_key=gemini_api_key)

def reset_csv_chat():
    keys_to_reset = [
        "chat_history_csv",
        "csv_file_content",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.chat_history_csv = [
        AIMessage(content="Hi there! How can I help you today?"),
    ]

def get_response(question, csv_file):
    agent = create_csv_agent(
        llm,
        csv_file,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True
    )
    response = agent.invoke(question)
    return response['output']


if "chat_history_csv" not in st.session_state:
    st.session_state.chat_history_csv = [
        AIMessage(content="Hi there! How can I help you today?"),
    ]

if "csv_file_content" not in st.session_state:
    st.session_state.csv_file_content = None


for message in st.session_state.chat_history_csv:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="bot.png"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="human.png"):
            st.write(message.content)


with st.sidebar:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read and store the file content
        csv_content = uploaded_file.read()
        st.session_state.csv_file_content = csv_content

    st.button(
    "🗑️ Reset conversation", 
    on_click=reset_csv_chat,
    use_container_width=True
    )

if st.session_state.csv_file_content is not None:
    # Create a BytesIO object from the stored content
    csv_file = io.BytesIO(st.session_state.csv_file_content)
    
    if user_query := st.chat_input("Ask questions about your CSV file"):
        st.session_state.chat_history_csv.append(HumanMessage(content=user_query))
        with st.chat_message("Human", avatar="human.png"):
            st.write(user_query)
        with st.spinner("Thinking..."):
          try:
            response = get_response(user_query, csv_file)
            
            with st.chat_message("AI", avatar="bot.png"):
                st.write(response)
              
          except Exception as e:
            st.write(f"An Error has occurred \n\n{e}")
            
        st.session_state.chat_history_csv.append(AIMessage(content=response))
else:
    st.info("Upload a CSV file to get started.")
