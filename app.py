import numpy as np
from flask import Flask, request
from flask import render_template
import sklearn
import pickle


app = Flask(__name__)
lr=pickle.load(open('data.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]

    prediction = lr.predict(final)[0]
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
