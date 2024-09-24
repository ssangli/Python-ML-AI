from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid

###### LLM #####

class LLMInterface:
    def __init__(self, llm):
        self.prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI bot. You are going to find relavent information from the uploaded documents"), MessagesPlaceholder(
            variable_name="chat_history"), ("human", "{input}"), ])
        self.llm = llm
        self.chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_message_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        self.message_store = {}

    def get_message_history(self, session_id):
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        return self.message_store[session_id]

    def generate_session_id(self):
         myuuid = uuid.uuid4()
         return str(myuuid)

    def generate_summary(self, data):
        template = """ You are an excellent research paper summarizer. Summarize paper {data} and show highlights and important equations and data tables """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        response = chain.invoke({"data" : data})
        return response

    def get_llm_response(self, user_message, docs, state):
        """
            Main function getting LLM response
        """
        print("State {} ".format(state))
        if state["start_session"] is False:
            state["start_session"] = True
            state["id"] = self.generate_session_id()
        data = ""
        for d in docs:
            data += d.page_content
        user_message = data + " " + user_message
        response = self.chain_with_history.invoke(
            {"input": user_message},
            {"configurable": {"session_id": state["id"]}}
        )
        print("Response {}".format(response))
        state["history"].append({"role" : "assistant",
                                 "content" : response})

