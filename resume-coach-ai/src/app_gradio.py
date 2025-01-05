from langchain_community.llms import Ollama
from utils.llm import LLMInterface
from utils.upload_file import UploadFile
import gradio as gr
import uuid

# Globals
llm = Ollama(model="llama3", base_url="http://127.0.0.1:11434")
llm_interface = LLMInterface(llm)

###########3 Chatbot ##################

file_uploader = UploadFile()
def response(message : str, chat_history : list, filename, job_desc, state ):
    bot_response, state = llm_interface.get_llm_cot_response(message, chat_history, resume_text, job_desc, state)
    chat_history.append((message, bot_response))
    return "", chat_history, state

# Function receiving job description
def get_job_description(data: str):
    return data

# File Upload
def upload_file1(filename):
    resume_text = file_uploader.get_text(filename)
    return resume_text

def generate_session_id():
    myuuid = uuid.uuid4()
    return str(myuuid)

def gen_cover_letter(resume_text, job_descr):
    response = llm.write_cover_letters(resume_text, job_descr)
    return response

# UI




# if __name__ == '__main__':
state = gr.State({"analyze_resume_and_job_description" : False, "id" : "", "resume_text" : ""})
with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Resume Coach")
    gr.Markdown("## Step 1: Upload resume(.pdf) and enter the job description")
    with gr.Row():
        with gr.Column():
            resume_file = gr.Files(file_count='single', type='filepath', label="Resume", container=True, )
            resume_file.upload(upload_file1, [ resume_file ], [resume_text])
            state["resume_text"] = resume_text
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
    chat_msg = gr.Textbox()
    chat_msg.submit(fn= response, inputs=[chat_msg, chatbot, resume_text, job_descr, state], outputs=[chat_msg, chatbot, state])
    cov_letter_btn = gr.Button("Generate cover letter")
    cov_letter_btn.click(gen_cover_letter, inputs=[resume_text, job_descr], output=[cover_letter])

gr.close_all()
demo.launch(share=True)
