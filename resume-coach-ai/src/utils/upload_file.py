from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from .llm import LLMInterface

class UploadFile:
    """
    Utility for handling pdf file upload
    """

    @staticmethod
    def process_uploaded_file(filename):
        print("Filename ", filename)
        if filename is None or len(filename) == 0:
            return None
        loader = PyMuPDFLoader(filename)
        text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separator="\n",
                )
        data = loader.load_and_split(text_splitter=text_splitter)
        text = ""
        for i in range(len(data)):
            text += data[i].page_content
        return text


