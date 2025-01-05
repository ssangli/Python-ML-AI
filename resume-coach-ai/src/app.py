from langchain_community.llms import Ollama
from utils.llm import LLMInterface
from utils.upload_file import UploadFile
import gradio as gr
import uuid
import yaml
import streamlit as st
import io, PyPDF2
from io import StringIO

# Globals
llm = Ollama(model="llama3", base_url="http://127.0.0.1:11434")
llm_interface = LLMInterface(llm)
#######################################
if "history" not in st.session_state:
    st.session_state["history"] = []

if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""

if "id" not in st.session_state:
    st.session_state["id"] = ""

if "job_descr" not in st.session_state:
    st.session_state["job_descr"] = ""

if "analyze_resume_and_job_description" not in st.session_state:
    st.session_state["analyze_resume_and_job_description"] = False

if "disable_file_upload" not in st.session_state:
    st.session_state["disable_file_upload"] = False

if "disable_job_descr" not in st.session_state:
    st.session_state["disable_job_descr"] = False
###########3 Chatbot ##################

def write_to_chat_window():
    for i in st.session_state["history"]:
        with st.chat_message(name=i["role"]):
            st.write(i["content"])

def write_to_text_area():
    if st.session_state['title'] is not None:
        st.markdown(st.session_state['title'])
    for i in st.session_state["summary"]:
        st.write(i["name"])
        st.write(i["content"])

file_uploader = UploadFile()
def respond(message : str, state : dict):
    llm_interface.get_llm_cot_response(message, state)

# Function receiving job description
def submit_job_descr():
    st.session_state["disable_job_descr"] = True
    st.session_state["job_descr"] = job_descr

# File Upload
def on_file_upload():
    #print("disable {}".format(st.session_state["disable_file_upload"]))
    st.session_state["disable_file_upload"] = True
    if uploaded_file is not None:
        pdf_file = io.BytesIO(uploaded_file.getvalue())
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        st.session_state["resume_text"] = resume_text

def generate_session_id():
    myuuid = uuid.uuid4()
    # Done only once per session
    if st.session_state["analyze_resume_and_job_description"] == False:
        st.session_state["id"] = myuuid
        st.session_state["analyze_resume_and_job_description"] = True

def gen_cover_letter():
    if st.session_state["resume_text"] is not None and st.session_state["job_descr"] is not None:
        response = llm_interface.generate_cover_letter(st.session_state["resume_text"], st.session_state["job_descr"])
        st.session_state["history"].append({"role" : "assistant", "content" : response})
        write_to_chat_window()
    else:
        st.write("Please provide resume and job description to proceed")

def clear_all_sidebar():
    st.session_state["disable_file_upload"] = False
    st.session_state["disable_job_descr"] = False

# UI
with st.sidebar:
    st.title("Upload Resume")
    uploaded_file = st.file_uploader(label="FileUpload", disabled = st.session_state["disable_file_upload"], accept_multiple_files = False, type=("pdf"), label_visibility="hidden", on_change=on_file_upload)
    if uploaded_file is not None:
        st.session_state["disable_file_upload"] = True
        pdf_file = io.BytesIO(uploaded_file.getvalue())
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        st.session_state["resume_text"] = resume_text
    st.title("Add job description")
    job_descr = st.text_area(label="Job Description", disabled=st.session_state["disable_job_descr"])
    st.button("Submit Job Description", on_click = submit_job_descr)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.button("Generate Cover Letter", on_click = gen_cover_letter)
        with col2:
            st.button("Clear all", on_click = clear_all_sidebar)

st.title("I am your resume coach. Upload Resume and provide job description and start chatting")
if __name__ == '__main__':
    if prompt := st.chat_input():
        generate_session_id()
        st.chat_message("user").write(prompt)
        st.session_state["history"].append({"role" : "user",
                                            "content" : prompt})
        respond(prompt,st.session_state)
        write_to_chat_window()
