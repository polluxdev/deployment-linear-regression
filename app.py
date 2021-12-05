import numpy as np
import json
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("apple_price_prediction.pkl")


@app.route('/', methods=['POST'])
def get_prediction():
    req_json = json.loads(request.data)  # read request
    data = [req_json['data']]
    x = np.array(data)
    result = model.predict(x)
    result_float = [float(i) for i in result]
    return jsonify({'output': result_float})  # return model output


app.run(port=6666, debug=True)
