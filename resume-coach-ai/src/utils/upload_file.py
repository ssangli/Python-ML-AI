from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from .llm import LLMInterface

class UploadFile:
    """
    Utility for handling pdf file upload
    """
    resume_text = ""

    def __init__(self):
        pass

    def _process_uploaded_file(self, filename):
        print("Filename ", filename)
        if filename is None or len(filename) == 0:
            return None
        loader = PyMuPDFLoader(filename)
        # not sure why text chunking is needed here
        """
        text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separator="\n",
                )
        data = loader.load_and_split(text_splitter=text_splitter)
        """
        for data in loader.load():
            self.resume_text += data.page_content

    def get_text(self, filename):
        self._process_uploaded_file(filename)
        return self.resume_text


