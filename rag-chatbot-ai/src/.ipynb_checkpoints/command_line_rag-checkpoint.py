import os,sys, yaml
import string
import nltk
from sklearn.feature_selection.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma

with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

def split_document(filename):
    text_splitter = CharacterTextSplitter(
            chunk_size=cfg['TEXT_SPLITTER']['chunk_size'],
            chunk_overlap=cfg['TEXT_SPLITTER']['chunk_overlap'],
            separator="\n",
            )
    loader = PyMuPDFLoader(filename)
    docs = loader.load_and_split()
    return docs

def process_document(data):
    punctuation = string.punctutation
    stopwords = stopwords.words('english')
    # first split by sentences
    sentences = []
    for sentence in sent_tokenize(data):
        mod_sentence = []
        for word in word_tokenize(sentence):
            word = word.lower()
            if not word.isAlpha() or word in stopwords or word in punctuation:
                continue
            mod_sentence.append(word)
        sentences.append(mod_sentence)
    return sentences

