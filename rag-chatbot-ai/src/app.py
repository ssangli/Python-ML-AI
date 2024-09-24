from langchain_community.llms import Ollama
import streamlit as st
import uuid
import yaml
import pprint
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import PyPDF2
from io import StringIO
from llm import LLMInterface

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
if "button_clicked" not in st.session_state:
    st.session_state["button_clicked"] = False
if "summary" not in st.session_state:
    st.session_state["summary"] = []

##################### DB ############################
def process_document(cfg, uploaded_files):
    data = []
    for f in uploaded_files:
        pdf_reader = PyPDF2.PdfReader(f)
        metadata = pdf_reader.metadata
        i = 1
        for page in pdf_reader.pages:
            data.append(Document(page_content=page.extract_text(), metadata = {'page' : i}))
            i += 1
    # create a text splitter
    text_splitter = CharacterTextSplitter(
            chunk_size=cfg['TEXT_SPLITTER']['chunk_size'],
            chunk_overlap=cfg['TEXT_SPLITTER']['chunk_overlap'],
            separator="\n",
            )
    docs = text_splitter.split_documents(data)
    embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2", model_kwargs = {"trust_remote_code":True})
    db = Chroma.from_documents(docs, embed_model, persist_directory=cfg['DB_PATH'])
    db.persist()
    return db
##################### DB ############################

def generate_session_id():
    myuuid = uuid.uuid4()
    return str(myuuid)

def respond(query : str, uploaded_files, state):
    print("messages {}, session state {}".format(query, state))
    db = process_document(cfg, uploaded_files)
    # Search for top 3 documents that match the query from the user
    docs = db.similarity_search(query, k = 3)
    print("Docs : ", docs)
    state["history"].append({"role" : "user",
                             "content" : query})
    #response = llm_interface.get_llm_response(query, state)
    llm_interface.get_llm_response(query, docs, state)
    print("State  in respond 2 ", state)

def read_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    data = ""
    for page in reader.pages:
        data += page.extract_text()
    return data

def summarizer_helper(uploaded_file):
        data = read_pdf(uploaded_file)
        response = llm_interface.generate_summary(data)
        st.session_state["summary"].append({"name" : uploaded_file.name,
                                                    "content" : response})

def summarize(uploaded_files):
    for f in uploaded_files:
        if len(st.session_state["summary"]) == 0:
            summarizer_helper(f)
        else:
            for i in st.session_state["summary"]:
                if "name" in i and i["name"] == f.name:
                    return
            summarizer_helper(f)

############################ STATE ###################

def click_action(uploaded_files):
    summarize(uploaded_files)
    for i in st.session_state["summary"]:
        st.write(i["name"])
        st.write(i["content"])

with st.sidebar:
    st.title("Upload Document")
    uploaded_files = st.file_uploader(label="FileUpload", accept_multiple_files = True, type=("pdf"), label_visibility="hidden")
    st.button("Summarize", on_click=click_action(uploaded_files))

st.title("Chat with your document")
if __name__ == "__main__":
    if prompt := st.chat_input():
        if len(uploaded_files) == 0:
            st.info("Please upload file to continue")
            st.stop()
    
        st.chat_message("user").write(prompt)
        respond(prompt, uploaded_files, st.session_state)
        for i in st.session_state["history"]:
            with st.chat_message(name=i["role"]):
                st.write(i["content"])

