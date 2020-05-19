import warnings
warnings.filterwarnings('ignore')
import numpy as np
from flask import Flask, request, jsonify, render_template, make_response
import pickle
import pandas as pd
import os
 
dirpath = os.getcwd()
app = Flask(__name__)
model = pickle.load(open(dirpath + '\\Forex_Predictor\\model\\model.pkl', 'rb'))

@app.route('/')
def home():
    '''
    For rendering results on HTML GUI
    '''
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(np.array(final_features))

    output = prediction
    if output == 0:
        output = 'Down'
    else:
        output = 'Up'

    return render_template('index.html', prediction_text='Your trading should be {}'.format(output))

@app.route('/preprocessing', methods = ['GET', 'POST'])
def preprocessing():
    file = request.files['datafile']
    if not file:
        text = 'No file selected'
    df = pd.read_csv(file)
    df['return_1'] = df['return_2'] - df['lag_return_1']
    df = df.to_html(dirpath+ '\\Forex_Predictor\\templates\\Pre-Processed_Data.html')
    text = 'preprocessed success'
    return render_template('Pre-Processed_Data.html')





@app.route('/predict_api',methods=['POST'])
def predict_api():
    ''' 
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)