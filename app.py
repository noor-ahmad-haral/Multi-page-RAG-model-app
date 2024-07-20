import streamlit as st


if "google_api_key" not in st.session_state:
    st.session_state["google_api_key"] = ""

st.logo("Untitled-1.png")
with st.sidebar:
    columns = st.columns([4,1], vertical_alignment="center")
    with columns[0]:
        with st.popover("Add Your API Key 🔧", use_container_width=True):
            google_api_key = st.text_input("Get API Key (https://aistudio.google.com/app/apikey)",
                                        value=st.session_state["google_api_key"],
                                        type="password")
        with columns[1]:
            clear_api_key = st.button("🗑️", type="primary", use_container_width=True)
            if clear_api_key:
                st.session_state["google_api_key"] = ""
                google_api_key = ""


if not google_api_key:
    st.info("Please Add Your API Key to proceed!")

else:
    st.session_state.google_api_key = google_api_key


    p1 = st.Page("ChatWithDocs.py", title="Chat With Docs", icon=":material/folder_open:")
    p2 = st.Page("ChatWithWeb.py", title="Chat With Web", icon=":material/language:")
    p3 = st.Page("ChatWithCSV.py", title="Chat With CSV", icon=":material/description:")
    p4 = st.Page("about.py", title="About", icon=":material/info:")

    pg = st.navigation({"Rag Models:":[p1, p2, p3, p4]})
    pg.run()
    