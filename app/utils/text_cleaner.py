import re 
import unicodedata
import nltk
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"RT", "", text)
    text = re.sub(r"via", "", text)
    text = re.sub(r"amp", "", text)
    text = re.sub(r"&amp;", "", text)
    return text.strip().lower()

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])



def preprocess_text(text: str) -> str:
    cleaned_text = clean_text(text)
    without_stopwords = remove_stopwords(cleaned_text)
    lemmatized_text = lemmatize_text(without_stopwords)
    return lemmatized_text
