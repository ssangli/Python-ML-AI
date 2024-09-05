from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid

###### LLM #####
class LLMInterface:
    def __init__(self, llm):
        self.prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI bot."), MessagesPlaceholder(
            variable_name="chat_history"), ("human", "{input}"), ])
        self.chain = self.prompt | llm
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

    def get_response_on_resume_and_job_description(self, resume_text, job_description, state):
        """
            This function will use llm to assess resume and job_description match. Currently, its not returing the response. But the response is stored in the history.
            TODO: the function can be modified to highlight matches and mismatches in keywords and provide a initial score.
                  Summarize the resume?
            This function will be called only once at the beginning.
        """
        print("Running evaluation on resume and job description")
        prompt = f"""Resume: {resume_text} \n End of resume. \n\n Job description: {job_description} \n End of job description. \n
            You are a world-class resume coach. Follow these steps to determine the match between the resume and the job description:
            1. From the resume and job description, identify the top 5 areas of expertise.
            2. List the areas of expertise from step 1 that are a match between the resume and the job description.
            3. Identify the areas of expertise in the job description that are not covered in the resume. Share them in the response.
            4. Based on the matches identified in step 2 and graps in step 3, generate a score between 0 (lowest) and 10 (highest) to respresent the amount of match between the resume and the job description.
            5. Come up with suggestions on how to improve the resume to cover the requirements in the job description.
            6. Ask me if you can help answer any questions that I have about your response.
            """
        self.prn_state(state)
        #state["id"] = self.generate_session_id()
        state["analyze_resume_and_job_description"] = True
        response = self.get_chain_with_history().invoke(
            {"input": prompt},
            {"configurable": {"session_id": state["id"]}}
        )
        self.prn_state(state)
        return response, state

    def get_llm_cot_response(self, user_message, chat_history, resume_text, job_description, state):
        """
            Main function getting LLM response with Chain of Thought Method
        """
        print("In llm cot response : session id ", state["id"])
        if state["id"] == "" or len(state["id"]) == 0:
            print("New Id is generated")
            state["id"] = self.generate_session_id()

        if resume_text is not None and job_description is not None and state["analyze_resume_and_job_description"] is False:
            response, state = self.get_response_on_resume_and_job_description(
                resume_text, job_description, state)
        else:
            response = self.get_chain_with_history().invoke(
                {"input": user_message},
                {"configurable": {"session_id": state["id"]}}
            )
        print(self.get_chain_with_history())
        self.prn_state(state)
        return response, state

    def prn_state(self, state):
        for k, v in state.items():
            print(k,v)

