import warnings
warnings.filterwarnings('ignore')
import numpy as np
from flask import Flask, request, jsonify, render_template, make_response
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('C:\\Users\\ToreLeon\\OneDrive\\Máy tính\\ Git\\Forex_Predictor\\model\\model.pkl', 'rb'))

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
    data = pd.read_csv(file)
    data['return_1'] = data['return_2'] - data['lag_return_1']
    data = data.to_csv('Pre-Processed_Data.csv', index = False)
    text = 'preprocessed success'
    return render_template('index.html', text = text, data = data)

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