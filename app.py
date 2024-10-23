from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    image_url = data.get('image_url')

    # <run model here>
    # currently working with request json payload:
    # {
    #     "data": "[www.google.com]"
    # }

    prediction = {"prediction": "Your Cat", "confidence": "95%"}

    return jsonify(prediction)



@app.route('/')
def home():
    return "Hello, Flask for WSL"
if __name__ == '__main__':
    app.run(debug=True)