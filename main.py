import nltk
from nltk.corpus.reader.tagged import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask

app = Flask(__name__)

@app.route("/")
def root():
    return "home"


@app.route("/emotion/<text>")
def predict_emotion(text):
    with open("emotion_modelo.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    with open("most_common_tokens.pkl", "rb") as file:
        loaded_tokens = pickle.load(file)

    nltk.download('punkt')
    nltk.download("stopwords")
    nltk.download("names")

    lemmatizer = WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words("english")
    names = nltk.corpus.names.words()

    def obtener_tokens_para_prediccion(text):

        tokens = word_tokenize(text)
        cleaned_tokens = []

        for token in tokens:
          if token in stopwords: continue
          if token in names: continue
          if not token.isalpha(): continue
          token = token.lower()
          token = lemmatizer.lemmatize(token)
          if token not in loaded_tokens: continue
          cleaned_tokens.append(token)

        return " ".join(cleaned_tokens)
    
    tokens_limpios = obtener_tokens_para_prediccion(text)
    
    vectorizador = TfidfVectorizer(vocabulary=loaded_tokens)
    X = vectorizador.fit_transform([tokens_limpios])

    classification = loaded_model.predict(X.toarray())

    return str(classification[0])


@app.route("/positivity/<text>")
def predict_positivity(text):
    with open("positivity_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    with open("most_common_tokens_positivity.pkl", "rb") as file:
        loaded_tokens = pickle.load(file)


    nltk.download('punkt')
    nltk.download("stopwords")
    nltk.download("names")

    lemmatizer = WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words("english")
    names = nltk.corpus.names.words()

    def obtener_tokens_para_prediccion(text):

        tokens = word_tokenize(text)
        cleaned_tokens = []

        for token in tokens:
          if token in stopwords: continue
          if token in names: continue
          if not token.isalpha(): continue
          token = token.lower()
          token = lemmatizer.lemmatize(token)
          if token not in loaded_tokens: continue
          cleaned_tokens.append(token)

        return " ".join(cleaned_tokens)
    
    tokens_limpios = obtener_tokens_para_prediccion(text)
    
    vectorizador = TfidfVectorizer(vocabulary=loaded_tokens)
    X = vectorizador.fit_transform([tokens_limpios])

    classification = loaded_model.predict(X.toarray())

    return  str(classification[0])

if __name__ == "__main__":
    app.run(debug = True)

