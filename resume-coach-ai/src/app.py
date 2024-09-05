from langchain_community.llms import Ollama
from utils.llm import LLMInterface
from utils.upload_file import UploadFile
import gradio as gr
import uuid

# Globals
llm = Ollama(model="llama3", base_url="http://127.0.0.1:11434")
llm_interface = LLMInterface(llm)

###########3 Chatbot ##################

def response(message : str, chat_history : list, filename, job_desc, state ):
    resume_text = UploadFile.process_uploaded_file(filename)
    bot_response, state = llm_interface.get_llm_cot_response(message, chat_history, resume_text, job_desc, state)
    chat_history.append((message, bot_response))
    return "", chat_history, state

# Function receiving job description
def get_job_description(data: str):
    return data

# File Upload
def upload_file1(filename):
    return filename

def generate_session_id():
    myuuid = uuid.uuid4()
    return str(myuuid)

# UI
# if __name__ == '__main__':
with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Resume Coach")
    gr.Markdown("## Step 1: Upload resume(.pdf) and enter the job description")
    with gr.Row():
        with gr.Column():
            resume_file = gr.Files(file_count='single', type='filepath', label="Resume", container=True, )
            resume_file.upload(upload_file1, [ resume_file ], [resume_file])
        with gr.Column():
            job_descr = gr.Textbox(label="Job Description", show_label=False, placeholder="Enter Job Description here", lines=14, max_lines=14)
            with gr.Row(equal_height=True):
                submit_btn = gr.Button("Submit")
                clear = gr.Button("Clear")
            submit_btn.click(get_job_description, inputs=[job_descr], outputs=[job_descr])
    with gr.Row():
        pass
    gr.Markdown("## Step 2: Start chatting...")
    chatbot = gr.Chatbot()
    state = gr.State({"analyze_resume_and_job_description" : False, "id" : ""})
    chat_msg = gr.Textbox()
    chat_msg.submit(fn= response, inputs=[chat_msg, chatbot, resume_file, job_descr, state], outputs=[chat_msg, chatbot, state])

gr.close_all()
demo.launch(share=True)
