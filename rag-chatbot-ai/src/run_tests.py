import command_line_rag
from text_processing import TextPreprocess

url = "https://arxiv.org/pdf/2308.16505"
data = command_line_rag.read_file(url)
data_str = ""
for d in data:
    data_str += d.page_content
preprocessed_data = TextPreprocess.preprocess_data(data_str)
command_line_rag.extract_keywords(preprocessed_data)
