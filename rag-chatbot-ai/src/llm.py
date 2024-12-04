from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid, re
import ragUtil

###### LLM #####

class LLMInterface:
    def __init__(self, llm):
        self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "you are an helpful AI bot. Answer the query to the best of your ability using the given {context} only. If you dont know the answer, please say so"),
                    MessagesPlaceholder(variable_name = "chat_history"),
                    ("human", "{query}")
                ]
        )
        self.llm = llm
        self.chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_message_history,
            input_messages_key="query",
            history_messages_key="chat_history",
        )
        self.message_store = {}

    def get_message_history(self, session_id):
        if session_id not in self.message_store:
            self.message_store[session_id] = ChatMessageHistory()
        print("ChatMessageHistory for session_id {}".format(session_id))
        print(self.message_store[session_id])
        return self.message_store[session_id]

    def generate_session_id(self):
         myuuid = uuid.uuid4()
         return str(myuuid)

    def generate_summary(self, embed_generator):
        print("----------------------------------")
        print("Generating Summary.......")
        template = """ You are an excellent research paper summarizer. Summarize paper {context} and show highlights and important equations and data tables. You keep all the important information in the summary.
        You also suggest top 3 key questions that the document answers to"""
        context = embed_generator.generate_context_for_summary_using_pagerank()
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        response = chain.invoke({"context" : context})
        print("Summary........")
        print(response)
        return response

    def get_context(self, user_message, state):
        context = ""
        for embed_gen in state["embed_generators"]:
            context += embed_gen.generate_context(user_message)
        return context

    def generate_questions_from_doc(self, embed_generator):
        template = """You are an excellent document analyzer. Analyze the following {data} and extract top 5 questions from it"""
        data = embed_generator.get_all_data()
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        response = chain.invoke({"data" : data})
        print("Queries ")
        print(response)

    def get_llm_response(self, user_message, state):
        """
            Main function getting LLM response
        """
        print("State {} ".format(state))

        context = self.get_context(user_message, state)
        print("-----------Context--------------")
        print(context)
        print("-----------Query----------------")
        query = f"Answer the query {user_message} using only the context {context} provided here. If the {context} is not relevant, please say so."
        print(query)
        response = self.chain_with_history.invoke(
            {"context" : context, "query": user_message},
            {"configurable": {"session_id": state["id"]}}
        )
        print("Response {}".format(response))
        state["history"].append({"role" : "assistant",
                             "content" : response})
