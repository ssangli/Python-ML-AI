from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import gradio as gr

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

###### LLM #####

class LLMInterface:
    def __init__(self):
        self.llm = Ollama(model="llama3")
        self.prompt = ChatPromptTemplate.from_messages([ ("system", "You are a helpful AI bot."), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}"), ])
        self.chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_message_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        self.message_store = {}

    def get_chain_with_history(self):
        return self.chain_with_history

    def get_message_history(self, session_id):
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        return self.message_store[session_id]

    def get_llm_response(self, user_message, chat_history, resume_text, job_description, session_id):
        print("session id type {}, session id {}".format(type(session_id), ''.join(session_id)))
        resume_jd = f"resume :{resume_text}, job_description: {job_description}"
        response = self.get_chain_with_history().invoke(
                {"input":resume_jd},
                {"configurable" : {"session_id": ''.join(session_id)}}
                )
        return response

############ Utility #############################
class UploadFile:
    @staticmethod
    def process_uploaded_file(filename):
        loader = PyMuPDFLoader(filename)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        data = loader.load_and_split(text_splitter=text_splitter)
        return data

###########3 Chatbot ##################

def response(message : str, chat_history : list, session_id, filename: str, job_desc: str):
    print("session id {}, resume {}".format(session_id, filename))
    resume_text = UploadFile.process_uploaded_file(filename)
    bot_response = llm_interface.get_llm_response(message, chat_history, resume_text, job_desc, session_id)
    chat_history.append((message, bot_response))
    return "", chat_history, session_id

# Function receiving job description
def get_job_description(data: str, session_id):
    return gr.Textbox(interactive=False), session_id

# File Upload
def upload_file(filename, session_id):
    #print("session id ", session_id)
    if filename is None:
        return session_id
    #print("Message type {}, data {}".format(type(filename), filename))
    return session_id

# UI
with gr.Blocks() as demo:
    gr.Markdown("Upload resume(.pdf) and enter the job description")
    session_id = gr.State(["foo"])
    with gr.Row():
        with gr.Column():
            files = gr.File(file_count="single", type='filepath')
            multimodal = gr.MultimodalTextbox(interactive=True, placeholder="Upload a File", file_types=['.pdf'])
            multimodal.submit(upload_file, [files, session_id], [session_id])
        with gr.Column():
            input_msg = gr.Textbox(label="Job Description", placeholder="Enter Job Description here as text or as URL")
            submit_btn = gr.Button("Submit")
            clear = gr.Button("Clear")
            submit_btn.click(get_job_description, inputs=[input_msg, session_id], outputs=[input_msg, session_id])
    chatbot = gr.Chatbot()
    chat_msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_msg.submit(fn= response, inputs=[chat_msg, chatbot, session_id, files, input_msg], outputs=[chat_msg, chatbot, session_id])

gr.close_all()

if __name__ == '__main__':
    llm_interface = LLMInterface()
    demo.launch(share=True)



