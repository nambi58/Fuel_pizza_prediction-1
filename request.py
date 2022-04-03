import requests

url = 'http://localhost:5000/predict_api' 
r = requests.post(url,json={'age':20, 'weight':40 , 'distance':20 })

print(r.json())


