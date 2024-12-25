import string
import nltk
import re
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocess:

    @staticmethod
    def preprocess_data(data):
        """
            Remove digits
            remove links
            convert to lower case
            remove stop words
            convert to stem stop
            get keywords using tfidf/bm25
        """
        stopwords = nltk.corpus.stopwords.words('english')
        stemmer = PorterStemmer()
        tokenizer = WhitespaceTokenizer()
        punct = string.punctuation
        pat = re.compile(r"http\S+|\\S+|\d+|\<\S+|\/|\>|\!|\,|\.")
        pat1 = re.compile(r'\x03|\x0f|\x11|\x14|\x15|\x17|\x16')
        data = re.sub(pat, "", data)
        data = re.sub(r' +', ' ', data)
        mod_data = []
        stem_word_map = {}
        for word in word_tokenize(data):
            word = word.lower()
            if word not in stopwords and word not in punct and re.match(pat, word) is None:
                #stem_word = stemmer.stem(word)
                #stem_word_map[word] = stem_word
                mod_data.append(word)
        return ' '.join(mod_data)
