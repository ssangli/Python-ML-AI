from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
import requests, io, PyPDF2, re, sys, yaml
import numpy as np
import networkx as nx

import text_processing

with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

embed_model = SentenceTransformer(cfg['EMBED_MODEL'])

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


class EmbeddingGenerator:
    _data_chunks = []
    _embeddings = np.array([])
    _title = None
    _keywords = []

    def __init__(self, filename):
        print("filename : ", filename)
        pdf_url = re.search(r"http.*", filename)
        data = ""
        if pdf_url is not None:
            data = self.read_pdf_url(pdf_url[0])
        else:
            data = self.read_file(filename)
        # Create embedding with data
        print("Title : {}".format(self._title))
        if self._title is not None:
            data = "Title : " + self._title + data
        # extract keywords
        self._extract_keywords(data)
        self.split_document_with_charsplitter(data)
        self.create_embeddings()

    def read_file(self, filename):
        loader = PyMuPDFLoader(filename)
        # extract_title
        try:
            first_page = loader.load()[0]
            if first_page.metadata and 'title' in first_page.metadata and first_page.metadata['title'] is not None:
                self._title = first_page.metadata['title']
            else:
                text = first_page.page_content
                till_abstract = text.split('Abstract')[0]
                self._title = ' '.join(till_abstract.split("\n"))
        except Exception as e:
            print(f"Error extracting title : {e}")
            self._title = None
        data = ""
        for p in loader.load():
            data += p.page_content
        return data

    def _extract_title(self, pdf_reader):
        try:
            if pdf_reader.metadata and 'Title' in pdf_reader.metadata:
                self._title = pdf_reader.metadata['Title']
            else:
                if len(pdf_reader.pages) > 0:
                    page = pdf_reader.pages[0]
                    text = page.extract_text()
                    first_line = text.split('Abstract')[0].split('\n')
                    self._title = ' '.join(first_line)
        except Exception as e:
            print(f"Error extracting title : {e}")
            self._title = None

    def read_pdf_url(self, url):
        response = requests.get(url)
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        self._title = self._extract_title(pdf_reader)
        data = ""
        for d in pdf_reader.pages:
            data += d.extract_text()
        return data

    # Character based chunking
    def split_document_with_charsplitter(self, data):
        text_splitter = CharacterTextSplitter(
                chunk_size=cfg['TEXT_SPLITTER']['chunk_size'],
                chunk_overlap=cfg['TEXT_SPLITTER']['chunk_overlap'],
                separator="\n",
                )
        self._data_chunks = text_splitter.split_text(data)
        print("-------------Data Chunks --------------")
        print(self._data_chunks[:5])

    # split document on sentences
    def split_document_on_sentence(self, data):
        print("Splitting document based on sentences.....")
        #data = read_file(filename)
        for d in data:
            content = d.page_content
            for sent in sent_tokenize(content):
                self._data_chunks.append(sent)

    def create_embeddings(self):
        embeddings = []
        for d in self._data_chunks:
            embeddings.append(embed_model.encode(d))
        self._embeddings = np.array(embeddings)

    def get_cosine_similarity(self, emb1, emb2):
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        return  dot_product / (norm_a * norm_b)

    def cosine_similarity(self, query_embeddings):
        cosine_similarity ={}
        for i, emb in enumerate(self._embeddings):
            cosine_similarity[i] = self.get_cosine_similarity(emb, query_embeddings)
        return cosine_similarity

    def euclidean_similarity(self, query_embeddings):
        euc_dist = {}
        for i, emb in enumerate(self._embeddings):
            euc_dist[i] = np.linalg.norm(emb - query_embeddings)
        return euc_dist

    def similarity_search(self, query_embeddings, type="cosine", top_k = 3):
        if type=="cosine":
            result = self.cosine_similarity(query_embeddings)
            top_k_sorted = sorted(result.items(), key = lambda x : x[1], reverse=True)[:top_k]
        else:
            result = self.euclidean_similarity(query_embeddings)
            top_k_sorted = sorted(result.items(), key=lambda x : x[1])[:top_k]
        return [x[0] for x in top_k_sorted]

    def generate_context(self, query, search_type="cosine", top_k=5):
        query_embeddings = np.array(embed_model.encode(query))
        similar_doc_indices = [self.similarity_search(query_embeddings, search_type, top_k)]
        context = []
        for idx in similar_doc_indices:
            context.append(self._data_chunks[idx[0]])
        return ' '.join(context)

    def generate_context_for_summary_using_pagerank(self):
        # generate similarity_matrix
        sim_matrix = np.zeros([len(self._data_chunks), len(self._data_chunks)])
        for i in range(len(self._data_chunks)):
            for j in range(len(self._data_chunks)):
                if i != j:
                    sim_matrix[i][j] = self.get_cosine_similarity(self.
                    _embeddings[i], self._embeddings[j])
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        ranked_data = sorted(((scores[i], s) for i, s in enumerate(self._data_chunks)), reverse=True)
        context = ""
        for i in range(10):
            context += ranked_data[i][1]
        return context

    def get_title(self):
        return self._title

    def get_all_data(self):
        return ' '.join(self._data_chunks)

    def get_embeddings(self):
        return self._embeddings

    def process_document(self, data):
        punctuation = string.punctuation
        stop_words = stopwords.words('english')
        # first split by sentences
        sentences = []
        for sentence in sent_tokenize(data):
            mod_sentence = []
            for word in word_tokenize(sentence):
                word = word.lower()
                if not word.isalpha() or word in stop_words or word in punctuation:
                    continue
                mod_sentence.append(word)
            sentences.append(' '.join(mod_sentence))
        return sentences

    def _extract_keywords(self, data):
        mod_data = text_processing.preprocess_tfidf(data)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([mod_data])
        _keywords = matrix.get_feature_names_out()

    def get_keywords(self):
        return _keywords[:7]
