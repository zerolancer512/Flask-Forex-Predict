import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from flask import Flask, request, jsonify, render_template, make_response, json
import pickle
import pandas as pd
import os
from xgboost import XGBClassifier, Booster
from scipy import stats


dirpath = os.getcwd()
app = Flask(__name__)


model = XGBClassifier()
model.load_model('model.pkl')


@app.route('/')
def home():
    '''
    For rendering results on HTML GUI
    '''
    return render_template('index.html')

@app.route('/download/<path:filename>', methods = ['GET', 'POST'])
def download(data, filename):
    resp = make_response(data.to_csv(index = False))
    resp.headers["Content-Disposition"] = "attachment; filename={}.csv".format(filename)
    resp.headers["Content-Type"] = "text/csv"
    return resp



@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
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
        return render_template('index.html', message_1 = 'No file selected')
    else:
        df = pd.read_csv(file)
        z_scores = stats.zscore(df)
        abs_z_scores = np.abs(z_scores)
        df = df[(abs_z_scores < 4).all(axis=1)]
        df['return_1'] = df['return_2'] - df['lag_return_1']
        render_template('index.html', message_1 = 'Pre - processing completed')
        return download(df, 'Preprocessed_Data')
        
        
@app.route('/FeatureSelection', methods = ['GET', 'POST'])
def FeatureSelection():
    file = request.files['datafile']
    if not file:
        return render_template('index.html', message_2 = 'No file selected')
    else:
        df = pd.read_csv(file)
        if 'up_down' not in df.columns:
            df = df[['Close', 'High', 'Low', 'return_1', 'return_2', 'return_3']]
        df = df[['Close', 'High', 'Low', 'return_1', 'return_2', 'return_3', 'up_down']]
        render_template('index.html', message_2 = 'Feature Selection completed')
        return download(df, 'FeatureSelected_Data')

@app.route('/TrainNewData', methods = ['GET', 'POST'])
def TrainNewData():
    file = request.files['datafile']
    if not file:
        return render_template('index.html', message_6 = 'No file selected')
    else:
        df = pd.read_csv(file)
        X = df[['Close', 'High', 'Low', 'return_1', 'return_2', 'return_3']]
        y = df['up_down']
        model.fit(X.values,y, xgb_model = 'model.pkl')
        model.save_model('model.pkl')
        return render_template('index.html', message_6 = 'Training completed')



@app.route('/Classification', methods = ['GET', 'POST'])
def Classification():
    file = request.files['datafile']
    if not file:
        return render_template('index.html', message_3 = 'No file selected')
    else:
        df = pd.read_csv(file)
        X_test = df[['Close', 'High', 'Low', 'return_1', 'return_2', 'return_3']]
        df['Up_down_predict'] = model.predict(X_test.values)
        render_template('index.html', message_3 = 'Classification completed')
        return download(df, 'Predicted_Data')

@app.route('/Evaluation', methods = ['GET', 'POST'])
def Evaluation():
    file = request.files['datafile']
    if not file:
        return render_template('index.html', message_4 = 'No file selected')
    else:
        df = pd.read_csv(file)
        actual_label = df['up_down']
        predict_label = df['Up_down_predict']
        Precision_Score = round(precision_score(actual_label, predict_label),4)
        Recall_Score = round(recall_score(actual_label, predict_label),4)
        F_score = round(f1_score(actual_label, predict_label),4) 
        return render_template('index.html', message_4 = "Precision Score: {}<br>Recall Score: {}<br>F1 Score: {}".format(Precision_Score, Recall_Score, F_score))

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