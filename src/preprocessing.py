import re
import nltk
from sklearn.datasets import fetch_20newsgroups

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def load_raw_data():
    return fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True, random_state=42
    )

def clean(text):
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[>|]{2,}', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def filter_documents(texts, labels, min_len=50, max_len=2000):
    out_t, out_l = [], []
    for t, l in zip(texts, labels):
        c = clean(t)
        if len(c) < min_len:
            continue
        out_t.append(c[:max_len])
        out_l.append(l)
    return out_t, out_l

def prepare_corpus():
    ds = load_raw_data()
    texts, labels = filter_documents(ds.data, ds.target)
    return texts, labels, ds.target_names
