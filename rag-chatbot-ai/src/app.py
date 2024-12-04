from langchain_community.llms import Ollama
import streamlit as st
import uuid, re, io, sys
import yaml
import pprint
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import PyPDF2
from io import StringIO
from llm import LLMInterface
import ragUtil

# Load the config
with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# Globals
llm = Ollama(model=cfg['LLM_MODEL'])
llm_interface = LLMInterface(llm)

############## STATE #####################################
if "id" not in st.session_state:
    st.session_state["id"] = ""
if "start_session" not in st.session_state:
    st.session_state["start_session"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []
if "pdf_urls" not in st.session_state:
    st.session_state["pdf_urls"] = []
if "summary" not in st.session_state:
    st.session_state["summary"] = []
if "embed_generators" not in st.session_state:
    st.session_state["embed_generators"] = []
if "file_embed_map" not in st.session_state:
    st.session_state["file_embed_map"] = {}
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = False
if "disable_pdf_url_link" not in st.session_state:
    st.session_state["disable_pdf_url_link"] = False
if "title" not in st.session_state:
    st.session_state["title"] = None

##################### DB ############################
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

def generate_session_id():
    myuuid = uuid.uuid4()
    return str(myuuid)

def respond(query : str, state : dict) -> str:
    print("messages {}, session state {}".format(query, state))
    # Extract the URL
    print("-------------Respond Module --------------")
    print(query)
    print("------------------------------------------")
    llm_interface.get_llm_response(query, state)

def summarizer():
    for filename, embed_gen in st.session_state["file_embed_map"].items():
        print("Filename {}".format(filename))
        response = llm_interface.generate_summary(embed_gen)
        title = embed_gen.get_title()
        st.session_state["summary"].append({"name" : filename, "content" :response})
        st.session_state["history"].append({"role" : "assistant", "content" : response})
        st.session_state["title"] = title
        write_to_text_area()

def summarizer_helper_old(uploaded_file):
    print("In summarize helper.....", uploaded_file)
    embed_generator = ragUtil.EmbeddingGenerator(uploaded_file)
    st.session_state["embed_generators"].append(embed_generator)
    st.session_state["file_embed_map"] = {uploaded_file: embed_generator}
    response = llm_interface.generate_summary(embed_generator)
    st.session_state["summary"].append({"name" : uploaded_file,
                                                "content" : response})
    st.session_state["history"].append({"role" : "assistant",
                                        "content" : response})
def summarize_old():
    if uploaded_file is not None:
        if len(st.session_state["summary"]) == 0:
            summarizer_helper(uploaded_file.name)
        else:
            for i in st.session_state["summary"]:
                if "name" in i and i["name"] == uploaded_file.name:
                    return
            summarizer_helper(uploaded_file.name)
    elif url_doc_link is not None:
        print("Document link {}".format(url_doc_link))
        summarizer_helper(url_doc_link)
    else:
        st.write("Please either upload a pdf or provide link to document")
    write_to_text_area()
############################ STATE ###################

def toggle():
    if url_doc_link is not None:
        st.session_state["disable_pdf_url_link"] = True

def clear():
    st.session_state["disable_pdf_url_link"] = False
    st.session_state["uploaded_file"] = False

def on_file_submit():
    if st.session_state["start_session"] is False:
        st.session_state["start_session"] = True
        st.session_state["id"] = generate_session_id()
        # Create new embed embed_generator
        if uploaded_file is not None:
            filename = uploaded_file.name
        else:
            filename = url_doc_link
        embed_gen = ragUtil.EmbeddingGenerator(filename)
        st.session_state["file_embed_map"] = {filename : embed_gen}
        # Extract title and key topics from this document
        st.session_state['title'] = embed_gen.get_title()
        write_to_text_area()


with st.sidebar:
    st.title("Upload Document")
    uploaded_file = st.file_uploader(label="FileUpload", accept_multiple_files = False, type=("pdf"), label_visibility="hidden", on_change=on_file_submit)
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = True
    st.title("OR")
    st.title("Provide URL link")
    url_doc_link = st.text_input(
        "Enter document URL 👇",
        disabled=st.session_state["disable_pdf_url_link"],
        on_change=toggle,
    )
    col1, col2, col3= st.columns(3)
    with col1:
        st.button("Submit", on_click= on_file_submit)
    with col2:
        st.button("Summarize", on_click = summarizer)
    with col3:
        st.button("Clear", on_click = clear)

st.title("Chat with your document")
if __name__ == "__main__":
    #st.write("Hi, I am your arxiv chatbot. Upload a PDF or a URL and start chatting with your document")
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        st.session_state["history"].append({"role" : "user",
                                            "content" : prompt})
        respond(prompt,st.session_state)
        write_to_chat_window()
