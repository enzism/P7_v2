import ast
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import uvicorn
import requests
import lightgbm

app = Flask(__name__)

model = joblib.load(open('model/LGBMClassifier.pkl', 'rb'))

@app.route('/predict', methods=['GET'])
def predict():
    PARAMS = request.args.get("data")
    PARAMS = PARAMS.replace(" ", ",")
    PARAMS = PARAMS.replace(",,", ",")
    PARAMS_to_array = ast.literal_eval(PARAMS)
    data = np.array(PARAMS_to_array)
    prediction = model.predict(data)
    proba = model.predict_proba(data)
    response = {"prediction": prediction[0]}
    return response

if __name__ == '__main__':
   app.run(debug=True)
