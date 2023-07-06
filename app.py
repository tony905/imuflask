from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('knn.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.525):
        return render_template('index.html',pred='Class = 1 probability for falling {}'.format(output))
    else:
        return render_template('index.html',pred='Class = 0 probability for falling {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
