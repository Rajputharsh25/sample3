# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

import sklearn
print(sklearn.__version__)



from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your model (Make sure the path is correct)
model = load('model.joblib')  # Assuming your model file is in the same directory as app.py

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Assuming your input is JSON or Form data
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])

    output = prediction[0]
    return render_template('index.html', prediction_text=f'Heart Disease Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

