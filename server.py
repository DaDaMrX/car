import flask

import api

app = flask.Flask(__name__)
nlu = api.HybridNLU(['rasa', 'rnn', 'ner'])


@app.route('/', methods=['GET'])
def index():
    return flask.redirect(flask.url_for('static', filename='index.html'))


@app.route('/api/nlu', methods=['POST'])
def api_nlu():
    data = flask.request.get_json()
    response = nlu.parse(data)
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=443, debug=False)
