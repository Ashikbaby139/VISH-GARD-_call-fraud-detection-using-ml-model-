import os
import pickle
import string
from functools import lru_cache


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

_tfidf = None
_model = None
_stopwords = None
_fraud_keywords = None
_normalized_fraud_keywords = None
_punctuation_table = str.maketrans("", "", string.punctuation)


def _load_artifacts():
    global _tfidf, _model

    if _tfidf is None:
        with open(VECTORIZER_PATH, "rb") as f:
            _tfidf = pickle.load(f)

    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)


from nltk.corpus import stopwords
import nltk

_stopwords = None

def _load_stopwords():
    global _stopwords

    if _stopwords is None:
        try:
            _stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            _stopwords = set(stopwords.words("english"))

    return _stopwords

def load_keywords(file_name="fraud_keywords.txt"):
    global _fraud_keywords, _normalized_fraud_keywords

    if _fraud_keywords is not None:
        return _fraud_keywords

    keywords = []
    file_path = os.path.join(BASE_DIR, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip().lower()
            if kw:
                keywords.append(kw)

    _fraud_keywords = tuple(keywords)
    _normalized_fraud_keywords = tuple((kw, kw.replace(" ", "")) for kw in _fraud_keywords)
    return _fraud_keywords


def preprocess(text):
    if not isinstance(text, str):
        return ""

    stopwords = _load_stopwords()
    cleaned = text.translate(_punctuation_table)
    words = [w for w in cleaned.split() if w.lower() not in stopwords]
    return " ".join(words)


def keyword_score(text):
    load_keywords()
    text_norm = "".join(str(text).lower().split())
    found = []

    for kw, kw_norm in _normalized_fraud_keywords:
        if kw_norm in text_norm:
            found.append(kw)

    return len(found), found


@lru_cache(maxsize=512)
def _predict_fraud_hybrid_cached(text):
    _load_artifacts()

    X = _tfidf.transform([preprocess(text)])
    proba = _model.predict_proba(X)[0]
    ml_confidence = proba[0] * 100

    kw_count, found_keywords = keyword_score(text)

    if "otp" in found_keywords:
        return 0, 100.0, found_keywords

    final_confidence = min(ml_confidence + (kw_count * 20), 100)
    decision = 0 if final_confidence >= 40 else 1

    return decision, round(final_confidence, 2), found_keywords


def predict_fraud_hybrid(text):
    safe_text = text if isinstance(text, str) else ""
    return _predict_fraud_hybrid_cached(safe_text)
