from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma

class VectorDB:
    @staticmethod
    def process_document(self, cfg, filenames):
        text_splitter = CharacterTextSplitter(
                chunk_size=cfg['TEXT_SPLITTER']['chunk_size'],
                chunk_overlap=cfg['TEXT_SPLITTER']['chunk_overlap'],
                separator="\n",
                )
        loaders = []
        for f in filenames:
            loaders.append(PyMuPDFLoader(file))
        docs = []
        for loader in loaders:
            docs.extend(loader.load_and_split(text_splitter=text_splitter))
        embeddings = cfg['EMBED_MODEL'].encode(docs)
        db = Chroma.from_documents(docs, embeddings, persistent_directory=cfg['DB_PATH'])
        return db



