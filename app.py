from flask import Flask, redirect, request, url_for, flash, send_file, Response
from jinja2 import Environment, PackageLoader
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np 

tokenizer = Tokenizer()

app = Flask(__name__)
get = Environment(loader=PackageLoader(__name__, 'templates')).get_template
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        band = request.form.get('Band', '')
        seed = request.form.get('seed', '')
        number = request.form.get('number', '')
        if band == 'Coldplay':
            data = open("coldplay/coldplay_lyrics.txt", encoding="utf8").read()
        else:
            data = open("twenty_one_pilots/top_lyrics.txt", encoding="utf8").read()
        dataset = data.lower().split("\n")
        for data in dataset:
            if data=='':
                dataset.remove('')
        tokenizer.fit_on_texts(dataset)
        total_words = len(tokenizer.word_index) + 1
        input_sequences = []
        for line in dataset:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequences = token_list[:i+1]
                input_sequences.append(n_gram_sequences)

        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        if band == 'Coldplay':
            model = tensorflow.keras.models.load_model('coldplay/coldplay_final_model_v1.h5')
        else:
            model = tensorflow.keras.models.load_model('twenty_one_pilots/top_model_v1.h5')
        seed_text=seed
        next_words=int(number)
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return get('lyrics_generator.html').render(answer=seed_text)
    return get('lyrics_generator.html').render()

if __name__ == '__main__':
    app.debug = True
    app.run()
