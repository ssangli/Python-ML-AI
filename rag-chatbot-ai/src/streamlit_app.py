from langchain_community.llms import Ollama
import llm
import ragUtil
import uuid
import streamlit as st

# Globals
llm_model = Ollama(model="llama3")
llm_session_id_map = {}

def generate_session_id():
    myuuid = uuid.uuid4()
    return str(myuuid)

def respond(messages : str, uploaded_file, job_descr, state):
    print("messages {}, session state {}".format(messages, state))
    print("Uploaded_file {}, job_descr {}".format(uploaded_file.name, job_descr))
    bot_response = llm_session_id_map[session_id].get_llm_response(message, chat_history)
    return messages

def response(message : str, state ):
    resume_file = ''.join(message['files'])
    session_id = ''.join(session_id)
    print("session id {}, resume {}".format(session_id, resume_file))
    if msg is None and resume_file is not None:
        llm_session_id_map[session_id].set_resume(resume_file)
        bot_response = llm_session_id_map[session_id].get_resume_summary()
    else:
        bot_response = llm_session_id_map[session_id].get_llm_response(message, chat_history)
    chat_history.append((msg, bot_response))
    return "", chat_history, gr.MultimodalTextbox(interactive=True, value=None)

# File Upload
def UploadDocs(filename, session_id):
    print("uploading file {} : uniq_id {} Sessionid {}".format(filename, uniq_id, session_id))
    print("filename ", filename)
    session_id = ''.join(session_id)
    llm_session_id_map.get(session_id).set_resume_file(filename)
    if filename is not None or len(filename) != 0:

        return session_id, gr.Textbox(value="File added to vector database. You can start chating with it now")
    else:
        return session_id, gr.Textbox(value="Upload Docs as .pdf")


with st.sidebar:
    st.title("Upload docs as .pdf")
    uploaded_file = st.file_uploader(label="UploadDocs", accept_multiple_files = False, type=("pdf"), label_visibility="hidden")

st.title("Welcome to Resume Coach")
if prompt := st.chat_input():
    if not uploaded_file:
        st.info("Please upload your Resume to continue.")
        st.stop()
    else:
        st.info("Thanks for uploading your resume")

    st.session_state = {}
    uuid = generate_session_id()
    st.session_state = ({"id" : uuid, "analyze_resume_and_job_description" : False})
    st.chat_message("user").write(prompt)
    response = respond(prompt, uploaded_file, job_desc, st.session_state)
    msg = response
    st.chat_message("resume-coach").write(msg)
