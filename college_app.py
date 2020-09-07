#!/usr/bin/env python
# coding: utf-8

# In[38]:


from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template("forest_fire.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
     int_features=[]
    for x in request.form.values():
        if(x==''):
            int_features.append(0)
        else:
            int_features.append(x)
    fina=np.array(int_features)
    for x in fina:
        if not x:
            x=0
    final=fina.reshape(-1,7)
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output=str(prediction[0])
    #output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.7):
        return render_template('forest_fire.html',pred='Congratulations !! Your Chances of Getting the university of your Choice are High.\nProbability of admission is {}'.format(output),bhai="Well done")
    else:
        return render_template('forest_fire.html',pred='Your Chances of getting the University of your Choice is little low but Dont Worry this prediction are Just based on previous datasets .\n Probability of admission  is {}'.format(output),bhai="Check another University")


if __name__ == '__main__':
    app.run(debug=False,port=1236)


# In[ ]:




