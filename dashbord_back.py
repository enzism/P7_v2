from fastapi import FastAPI
import pickle
import uvicorn

app = FastAPI()

model = pickle.load(open('model/LGBMClassifier.pkl', 'rb'))

@app.get("/")
def root():
    return {"message": "Loan Prediction"}

@app.get("/predict/")
def predict(sample):
    X = sample.iloc[:, :-1]
    score = model.predict_proba(X[X.index == int(194347)])[:, 1]
    return score

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)