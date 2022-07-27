from fastapi import FastAPI
from flask import Flask, request, jsonify
import pickle
import uvicorn
import requests
from dashboard_front import load_data

app = Flask(__name__)

model = pickle.load(open('model/LGBMClassifier.pkl', 'rb'))


@app.get("/")
def root():
    return {"message": "Loan Prediction"}

@app.get("/predict/")
def predict():
    id = requests.get('id')
    data, sample, target, description = load_data()
    X = sample[sample['SK_ID_CURR'] == id].to_numpy().tolist()
    prediction = model.predict(X)
    return {"message": f"Loan Prediction{prediction}"}

if __name__ == '__main__':
    app.run(debug=True)
    #uvicorn.run(app, host='127.0.0.1', port=8000)