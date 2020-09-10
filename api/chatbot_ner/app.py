from os.path import join, dirname
from ner_model import NameEntityRecognition

from flask import Flask, jsonify, render_template, request, url_for

app = Flask(__name__)

ner_model_path = join(dirname(__file__), "model/ner_model")
words_path = join(dirname(__file__), "model/data/words.pl")
embed_words_path = join(dirname(__file__), "model/data/vectors.npy")
tag_path = join(dirname(__file__), "model/data/tag_data.pkl")
data_path = join(dirname(__file__), "model/data")

params = [ner_model_path, words_path, embed_words_path, tag_path, data_path]
ner = NameEntityRecognition(*params)


@app.route('/name_entity', methods=['GET', 'POST'])
def get_name_entity():
    if request.method == 'POST':
        data = request.get_json()
        sentences = data["paragraph"]
        result = ner.predict(sentences)
        return {"result": result}


if __name__ == '__main__':

    app.run(host="0.0.0.0", port=11012, debug=True)