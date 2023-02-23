from flask import Flask, request, jsonify
from chat import get_answer

app = Flask(__name__)

@app.route('/chat', methods=['GET','POST'])
def chat():
    data = request.json
    response = jsonify(get_answer(data['sentence']))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)