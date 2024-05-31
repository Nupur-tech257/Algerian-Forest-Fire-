import pickle
from flask import Flask,request,app,jsonify
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model1.pkl','rb'))
model2=pickle.load(open('model2.pkl','rb'))
@app.route("/predictrg_api",methods=['POST'])
def predict_api():

    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    new_data= scaler.fit_transform(new_data)
    output=model.predict(new_data)[0]
    return jsonify(output)



@app.route("/predictcl_api",methods=['POST'])
def predictcl_api():

    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model2.predict(new_data)[0]
    return jsonify(int(output))
if __name__== "__main__":
    app.run(debug=True)