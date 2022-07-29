import ast

import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import uvicorn
import requests
from dashboard_front import load_data

app = Flask(__name__)

model = joblib.load(open('model/LGBMClassifier.pkl', 'rb'))

@app.route('/', methods=['GET'])
def predict():
    PARAMS_str = request.args.get("data")
    data = pd.DataFrame(ast.literal_eval(PARAMS_str)).to_numpy()
    prediction = model.predict(data)
    proba = model.predict_proba(data)
    response = {"prediction": prediction[0]}
    return response


if __name__ == '__main__':
   app.run(debug=True)

"""
@app.route('/predict', methods=['GET'])
def helloworld():
    return {"message": "Loan Prediction"}
    #if (request.method == 'GET'):
        id = request.args.get("data")
        #data = sample[sample.index == int(id)].to_numpy()[:,:-1]
        #prediction = model.predict(data)
        #response = {"prediction": prediction[0]}


if __name__ == '__main__':
    app.run(debug=True)
    #uvicorn.run(app, host='127.0.0.1', port=8000)

"""