from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid
import yaml

###############
###### LLM #####
class LLMInterface:
    system_prompt = ""
    with open("utils/prompt.yaml", "r") as f:
        system_prompt = yaml.safe_load(f)

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([("system", self.system_prompt["SYSTEM_PROMPT"]), MessagesPlaceholder(
            variable_name="history"), ("human", "{input}"), ])
        self.chain = self.prompt | llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_message_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        self.message_store = {}

    def get_chain_with_history(self):
        return self.chain_with_history

    def get_message_history(self, session_id):
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        return self.message_store[session_id]

    def generate_session_id(self):
         myuuid = uuid.uuid4()
         return str(myuuid)

    def get_resume_summary(self, resume_text, state):
        """
            Summarize the resume once the pdf is uploaded
        """
        print("Summarizing resume\n")
        summarize_resume = f"Could you summarize my resume?\n resume : {
            resume_text}"
        response = self.get_chain_with_history().invoke(
            {"input": summarize_resume},
            {"configurable": {"session_id": state["id"]}}
        )
        return response

    def get_job_description_summary(self, job_description, state):
        """
            List all the key skills from the job description
        """
        summarize_jd = f"Could you highlight key skill requirement in the job description?\n job_description: {
            job_description}"
        response = self.get_chain_with_history().invoke(
            {"input": summarize_jd},
            {"configurable": {"session_id": state["id"]}}
        )
        return response

    def get_llm_response(self, user_message, chat_history, resume_text, job_description, state):
        """
            Main function getting LLM response
        """
        if state["analyze_resume_and_job_description"] is False:
            response = self.get_response_on_resume_and_job_description(
                resume_text, job_description, state)
        response = self.get_chain_with_history().invoke(
            {"input": user_message},
            {"configurable": {"session_id": state["id"]}}
        )
        print(self.get_chain_with_history())
        return response, state


    # Chain of Thought Method
    def generate_cover_letter(self, resume_text, job_description):
        """
            Helps write cover letter matching the job description and resume
        """
        prompt_template = f"""
        You are an expert at writing impressive cover letters based on resume {resume_text} and job description {job_description}"
        Use only the content mentioned in {resume_text} and dont make up words.
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        response = chain.invoke({"resume_text" : resume_text, "job_description" : job_description})

        #response = self.get_chain_with_history().invoke(
        #        {"input" : prompt, "resume_text" : state["resume_text"], "job_description" : state["job_descr"]},
        #        {"configurable" : {"session_id" : state["id"]}},
        #        )
        return response

    def get_llm_cot_response(self, user_message, state):
        """
            Main function getting LLM response with Chain of Thought Method
        """
        print("In llm cot response : session id ", state["id"])
        print("Resume {}".format(state["resume_text"]))
        response = self.get_chain_with_history().invoke(
                    {"input": user_message, "resume_text" : state["resume_text"], "job_description" : state["job_descr"]},
                    {"configurable": {"session_id": state["id"]}}
                )
        print(self.get_chain_with_history())
        self.prn_state(state)
        state["history"].append({"role" : "assistant", "content": response})

    def prn_state(self, state):
        for k, v in state.items():
            print(k,v)
