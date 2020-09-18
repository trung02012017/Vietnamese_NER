from os.path import join, dirname, abspath
from ner_model import NameEntityRecognition
from money_parser import parse

from flask import Flask, jsonify, request, Response

import json

app = Flask(__name__)

root_dir = abspath(dirname(__file__))
ner_model_path = join(root_dir, "model/ner_model")
words_path = join(root_dir, "model/data/words.pl")
embed_words_path = join(root_dir, "model/data/vectors.npy")
tag_path = join(root_dir, "model/data/tag_data.pkl")
data_path = join(root_dir, "model/data")

params = [ner_model_path, words_path, embed_words_path, tag_path, data_path]
ner = NameEntityRecognition(*params)


@app.route('/name_entity', methods=['GET', 'POST'])
def get_name_entity():
    global ner
    if request.method == 'POST':
        data = request.get_json()
        sentences = data["paragraph"]
        result = ner.predict(sentences)
        return {"result": result}


@app.route('/ner_extra', methods=['POST'])
def process_request_ex():
    global ner
    try:
        data = request.data
        data = json.loads(data)
        print(u'Input:\n%s' % (data))
        try:
            result = ner.predict(data['data'], json_format=True)
            return jsonify(result)
        except:
            return Response(response='Service fail', status=500)
    except:
        return Response(response='Bad request', status=400)


@app.route('/parse_money', methods=['POST'])
def parse_money():
    try:
        data = request.data
        data = json.loads(data)
        print(u'Input:\n%s' % (data))
        try:
            result = parse(data['data'])
            return jsonify({'result': result})
        except:
            return Response(response='Service fail', status=500)
    except:
        return Response(response='Bad request', status=400)


if __name__ == '__main__':

    app.run(host="0.0.0.0", port=11012, debug=True)