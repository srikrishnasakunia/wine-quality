from re import template
from flask import Flask, render_template, request
import pickle
import numpy as np


model_lr = pickle.load(open('lr_model.pkl','rb'))
model_rf = pickle.load(open('rf_model.pkl','rb'))

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")
@app.route("/predict",methods = ['POST','GET'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final = [np.array(float_features)]
    prediction_lr = model_lr.predict(final)
    prediction_rf = model_rf.predict(final)
    
    # if((prediction_lr == 1) and (prediction_rf == 1)):
    #     return render_template('index.html',pred = "Your Wine is of Good Quality.")
    # elif (((prediction_lr == 0) and (prediction_rf == 1)) or ((prediction_lr == 1) and (prediction_rf == 0))):
    #     return render_template('index.html',pred = "Your Wine is of Okayish Quality.")
    # else:
    #     return render_template('index.html',pred = "Your Wine is of Bad Quality.")
    
    if((prediction_rf == 1)):
        return render_template('index.html',pred = "Your Wine is of Good Quality.")
    else:
        return render_template('index.html',pred = "Your Wine is of Bad Quality.")
if __name__ == "__main__":
    app.run(debug=True)