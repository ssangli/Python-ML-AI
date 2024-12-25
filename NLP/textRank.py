import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from text_processing import TextPreprocess
import nltk

class TextRank:
    def __init__(self, data, top_n, window_size, num_iters, d):
        nlp = spacy.load('en_core_web_sm')
        self.data = nlp(data)
        self.top_n = top_n
        self.window_size = window_size
        self.num_iters = num_iters
        self.d = d

    def chunk_data(self):
        processed_data = []
        stopwords = nltk.corpus.stopwords.words('english')
        candidate_pos = ["NOUN", "VERB", "PROPN"]
        punctuation = string.punctuation
        self.mod_sentence = []
        for sent in self.data.sents:
            mod_words = []
            for word in sent:
                if word.pos_ in candidate_pos and word.text.isalpha() and word.text not in punctuation and word.text not in stopwords:
                    mod_words.append(word.text.lower())
            self.mod_sentence.append(mod_words)

    def calculate_word_freq(self):
        word_freq = {}
        for sent in self.mod_sentence:
            for word in sent:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        return word_freq

    def get_window_list(self):
        words_in_window = {}
        for sent in self.mod_sentence:
            for i, w in enumerate(sent):
                for j in range(i+1, i+self.window_size):
                    if j >= len(sent):
                        break
                    wp = (sent[i], sent[j])
                    if wp not in words_in_window:
                        words_in_window[wp]  =1
                    else:
                        words_in_window[wp] += 1
        return words_in_window

    def generate_word_matrix(self, uniq_words, words_in_window):
        matrix = dict()
        for w1 in uniq_words:
            matrix[w1] = dict()
            for w2 in uniq_words:
                matrix[w1][w2] = 0
        # populate the matrix
        for w1 in uniq_words:
            for w2 in uniq_words:
                if (w1, w2) in words_in_window:
                    matrix[w1][w2] = words_in_window[(w1,w2)]
        return matrix

    def get_normalize_matrix(self, matrix):
        df = pd.DataFrame(matrix)
        np_arr = np.array(df)
        np_arr = np.where(np_arr == 0, 1e-10, np_arr)
        np_arr_norm = np_arr / np.sum(np_arr, axis =0, keepdims = True)
        return np_arr_norm

    def estimate_page_rank(self,norm_array, num_iters, norm_threshold, d):
        curr = np.ones((norm_array.shape[0],))
        while i in range(num_iters):
            curr = (1-d) + d * np.dot(norm_array, curr)
            norm = np.linalg.norm(curr)
            if norm <= norm_threshold:
                break
            i += 1

       return curr

    def get_keywords(self):
        self.chunk_data()
        word_freq = self.calculate_word_freq()
        words_in_window = self.get_window_list()
        uniq_words = list(word_freq.keys())
        matrix = self.generate_word_matrix(uniq_words, words_in_window)
        norm_array = self.get_normalize_matrix(matrix)
        rank_arr = self.estimate_page_rank(norm_array,self.num_iters, 1e-5, self.d )
        node_weight = dict()
        for i, w in enumerate(uniq_words):
            node_weight[w] = rank_arr[i]

        node_weight_sorted = sorted(node_weight, key = lambda x : x[1], reverse=True)
        return [k for k, v in node_weight_sorted.items()][:self.top_n]
