import os,sys, yaml
import string
import nltk, torch
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import numpy as np
#from sklearn.feature_selection.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from sklearn.cluster import KMeans
import networkx as nx
from keybert import KeyBERT
import spacy
#import pytextrank
from collections import Counter
from text_processing import TextPreprocess
from yake import KeywordExtractor

with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

llm = Ollama(model=cfg['LLM_MODEL'])
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

def read_file(filename):
    print("Reading file .....", filename)
    loader = PyMuPDFLoader(filename)
    return loader.load()

# Character based chunking
def split_document_char_based(filename):
    data = read_file(filename)
    text_splitter = CharacterTextSplitter(
            chunk_size=cfg['TEXT_SPLITTER']['chunk_size'],
            chunk_overlap=cfg['TEXT_SPLITTER']['chunk_overlap'],
            separator="\n",
            )
    docs = []
    for d in data:
        docs.append(d.page_content)
    return text_splitter.split_text(' '.join(docs))

# Sentence based chunking
def split_document_on_sentence(data):
    print("Splitting document based on sentences.....")
    #data = read_file(filename)
    docs = []
    for d in data:
        content = d.page_content
        for sent in sent_tokenize(content):
            docs.append(sent)
    return docs

def generate_embeddings(embed_model, data):
    #docs = split_document_char_based(filename)
    print("Generating Embeddings......")
    embeddings = []
    for d in data:
        embeddings.append(embed_model.encode(d))
    return torch.tensor(embeddings)

def cosine_similarity(embeddings, query_embeddings):
    cosine_similarity ={}
    for i, emb in enumerate(embeddings):
        dot_product = np.dot(emb, query_embeddings)
        norm_a = np.linalg.norm(emb)
        norm_b = np.linalg.norm(query_embeddings)
        cosine_similarity[i] = dot_product / (norm_a * norm_b)
    return cosine_similarity

def euclidean_similarity(embeddings, query_embeddings):
    euc_dist = {}
    for i, emb in enumerate(embeddings):
        euc_dist[i] = np.linalg.norm(emb - query_embeddings)
    return euc_dist

def similarity_search(embeddings, query_embeddings, type="cosine", top_k = 3):
    if type=="cosine":
        result = cosine_similarity(embeddings, query_embeddings)
        top_k_sorted = sorted(result.items(), key = lambda x : x[1], reverse=True)[:top_k]
    else:
        result = euclidean_similarity(embeddings, query_embeddings)
        top_k_sorted = sorted(result.items(), key=lambda x : x[1])[:top_k]
    return [x[0] for x in top_k_sorted]

def generate_context(data_chunks, embeddings, query_embeddings, search_type="cosine", top_k=3):
    similar_doc_indices = [similarity_search(embeddings, query_embeddings, search_type, top_k)]
    context = []
    for idx in similar_doc_indices:
        context.append(data_chunks[idx[0]])
    return ' '.join(context)

def cal_cosine_similarity(emb1, emb2):
    num = np.dot(emb1,emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    return num / (norm_a * norm_b)

def generate_context_for_summary_with_kmeans(docs, embeddings, num_clusters=8):
    # create sentence clusteres using kmeans
    print("Generating Clusters......")
    km = KMeans(n_clusters = num_clusters)
    km.fit_transform(embeddings)
    print("Generating context for summary.....")
    context = ""
    for c in km.cluster_centers_:
        dist = []
        for idx, emb in enumerate(embeddings):
            dist.append((np.sum(np.abs(emb.numpy()-c)), idx))
        topk = sorted(dist, key= lambda x : x[0])[:8]
        for v in topk:
            context += docs[v[1]]
    print("Context for Summary.....")
    print(context)
    return context

def generate_context_for_summary_using_pagerank(docs, embeddings):
    # generate similarity_matrix
    sim_matrix = np.zeros([len(docs), len(docs)])
    for i in range(len(docs)):
        for j in range(len(docs)):
            if i != j:
                sim_matrix[i][j] = cal_cosine_similarity(embeddings[i], embeddings[j])

    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_data = sorted(((scores[i], s) for i, s in enumerate(docs)), reverse=True)
    context = ""
    for i in range(10):
        context += ranked_data[i][1]
    return context

def generate_summary(data, embeddings):
    #docs = split_document_on_sentence(data)
    #embeddings = generate_embeddings(docs)
    # create cluster of sentences
    responses = []
    for generator in [generate_context_for_summary_using_pagerank]:
        context = generator(data, embeddings)
        #context = generate_context_for_summary(data, embeddings)
        prompt_template = """ You are an excellenet document summarizer. Summarize the document based on the context {context}"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({"context" : context})
        responses.append(response)
    return responses
"""
def extract_keywords_with_textrank(data):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    keywords = nlp(data)
    return keywords[:10]
    """

def extract_keywords_with_keybert(data):
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = model.extract_keywords(data)
    return keywords[:10]

def extract_keywords_with_spacy(data):
    nlp = spacy.load("en_core_web_sm")
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    keywords = []
    for d in data:
        if d in pos_tag:
            keywords.append(d)
    output = set(keywords)
    return Counter(output).most_common(10)

def extract_keywords_with_yake(data):
    kw_extractor = KeywordExtractor()
    keywords = kw_extractor.extract_keywords(data)
    return keywords[:10]

def extract_keywords(data):
    extract_funcs = {"keybert" : extract_keywords_with_keybert,
                     "spacy" : extract_keywords_with_spacy,
                     "yake" : extract_keywords_with_yake}
    for name, func in extract_funcs.items():
        top_10 = func(data)
        print(name)
        print(top_10)

if __name__ == '__main__':
    data = read_file(sys.argv[1])
    preprocessed_data = TextPreprocess.preprocess_data(data)
    extract_keywords(preprocessed_data)

    data_chunks = split_document_on_sentence(data)
    embeddings = generate_embeddings(embed_model, data_chunks)
    summary = generate_summary(data_chunks, embeddings)
    for s in summary:
        print("-------------Document Summary----------------")
        print(s)
