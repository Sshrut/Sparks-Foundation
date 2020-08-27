# -*- coding: utf-8 -*-
from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features=[float(x) for x in request.form.values()]
    int_feature=(np.array(int_features))
    final=int_feature.reshape(len(int_feature),1)
    prediction=round(model.predict(final)[0],2)
    
    return render_template('index.html', prediction_text='Predicted score of a\
                           student who studies {} hours of a day {} %'
                           .format(int_features[0],prediction))    


if __name__=='__main__':
    app.run(debug=True)