import os
import tempfile
import streamlit as st
from streamlit_chat import message
from agent import ChatPDF
import time
st.set_page_config(page_title="ChatPDF")

os.environ["DATA"] = "DATA/"
os.environ["LOAD"] = "LOAD/Documents/"

def stream_data(generate):
    for word in generate.split(" "):
        yield word + " "
        time.sleep(0.02)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    rag = st.toggle("Upload the File PDF into the vectorDb",value=False)

    st.markdown("## Setting Rag")
    if rag:
        st.markdown("##### Upload a document")
        file = st.file_uploader(
            "Upload document",
            type=["pdf"],
            key="file_upload",
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        for file in st.session_state["file_upload"]:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.spinner(f"Ingesting {file.name}"):
                    ChatPDF().ingest(file_path)
            os.remove(file_path)


        if st.button("Clear uploaded files"):
            if st.session_state["file_uploader"] is not None:
                st.session_state["file_uploader"] = None
                st.experimental_rerun()
            else:
                pass


if prompt := st.chat_input("Enter your query for the MSA Planner..."):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.chat_message("user"):
            st.markdown(prompt)

    with st.spinner("Processing your request..."):

        respond = ChatPDF().ask(prompt)
        print(respond)
        st.session_state.messages.append({"role": "assistant", "content": respond})
        if respond is not None: 
            with st.chat_message("assistant"):
                print(respond)
                st.markdown(st.write_stream(stream_data(respond)))



