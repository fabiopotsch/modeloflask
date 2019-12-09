from flask import Flask, request
import pickle
import sklearn.feature_extraction.text as vectorizers
import string
from string import digits
from nltk.corpus import stopwords

app = Flask('analise_sentimento')

@app.route("/analise_sentimento")
def hello_world():

    # Obtendo parâmetro.
    texto = request.args.get('texto')

    if texto is None:
        return "Você não informou o argumento necessário.", 404
    elif texto == '':
        return "O argumento requerido está vazio.", 200
    else:

        # Obtendo o classificador e o vocabulário do vetorizador.

        with open('../Modelos/modelo.pkl','rb') as f:
            classificador = pickle.load(f)

        with open('../Modelos/voabulario.pkl','rb') as f:
            vocabulario = pickle.load(f)

        # Instanciando o vetorizador.

        vetorizador = vectorizers.TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 1), binary=False,
                                                  vocabulary=vocabulario)

        # Tratamento do texto.
        # 1. Remoção de pontuação.
        # 2. Remoção de dígitos.
        # 3. Transformação do texto dos tweets para letras minúsculas.
        # 4. Separação de todas as palavras com espaços regulares.
        # 5. Remoção de stopwords.

        remove_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        remove_digits = str.maketrans(digits, ' ' * len(digits))

        texto = str(texto).translate(remove_punctuation)
        texto = str(texto).translate(remove_digits)
        texto = str(texto).lower()
        texto = ' '.join(str(texto).split())
        texto = ' '.join([word for word in str(texto).split(' ') if word not in stopwords.words('english')])

        # Vetorizando o texto tratado.

        vetorResultante = vetorizador.fit_transform([texto])

        # Classificando o texto vetorizado.

        sentimento = classificador.predict(vetorResultante)

        # Retornando sentimento predito.

        return 'O sentimento do seu texto é: ' + str(sentimento[0]), 200

app.run()