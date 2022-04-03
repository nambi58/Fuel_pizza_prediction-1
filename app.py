#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model_knn = pickle.load(open('model_knn.pkl', 'rb'))
model_lr = pickle.load(open('model_lr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pizza', methods=['POST', 'GET'])
def rpizza():
    return render_template('resultp.html')

@app.route('/fuel', methods=['POST', 'GET'])
def rfuel():
    return render_template('resultf.html')

@app.route('/resultp.html', methods=['POST', 'GET'])
def pizza():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_knn.predict(final_features)
    if prediction == 1:
        pred = "like Pizza"
    elif prediction == 0:
        pred = "don't like Pizza"
    output = pred
    return render_template('resultp.html', prediction_text='You {}'.format(output))

@app.route('/resultf.html', methods=['POST', 'GET'])
def fuel():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_lr.predict(final_features)
    output = prediction[0]
    return render_template('resultf.html', prediction_text='fuel price for kilometer driven is :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

