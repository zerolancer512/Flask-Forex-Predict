import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Close':2, 'High':9, 'Low':6, 'return_1':0, 'return_2':0, 'return_3':0})

print(r.json())