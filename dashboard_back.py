from fastapi import FastAPI
import ast
from flask import Flask, request, jsonify
import joblib
import uvicorn
import requests
from dashboard_front import load_data

app = Flask(__name__)

model = joblib.load(open('model/LGBMClassifier.pkl', 'rb'))
test, sample, target, description = load_data()

#@app.get("/")
#def root():
#    return {"message": "Loan Prediction"}


@app.route('/', methods=['GET'])
def helloworld():
    if (request.method == 'GET'):
        id = request.args.get("data")
        data = sample[sample.index == int(id)].to_numpy()[:,:-1]
        prediction = model.predict(data)
        response = {"prediction": prediction[0]}
        return response

if __name__ == '__main__':
    app.run(debug=True)

"""
@app.get("/predict")
def predict():
#    data = data.replace("(","[[")
#    data =data.replace(")","]]")
#    data = ast.literal_eval(data)
#    prediction = model.predict(data)
#    proba = model.predict_proba(data)
#    print(f"PREDICTION : {prediction}")
#    return {
#        'prediction': prediction[0],
#        'proba': proba
#    }
    id = request.args.get('data')
    return {"message": id}


    #uvicorn.run(app, host='127.0.0.1', port=8000)

"""
