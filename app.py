from flask import Flask, request
from flask import render_template
import pickle
import pandas
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        fixed_acidity = request.form['fixed acidity']
        volatile_acidity = request.form['volatile acidity']
        citric_acid = request.form['citric acid']
        chlorides = request.form['chlorides']
        density = request.form['density']
        pH = request.form['pH']
        sulphates = request.form['sulphates']
        alcohol = request.form['alcohol']
        type_white = request.form['type_white']
        best_quality = request.form['best quality']
        ls = [[float(fixed_acidity), float(volatile_acidity), float(citric_acid),
               float(chlorides), float(density), float(pH),
               float(sulphates), float(alcohol), int(type_white), int(best_quality), ]]
        lr = pickle.load(open('data2.pkl', 'rb'))
        prediction=lr.predict(ls)[0]
        #prediction = lr[0].predict(ls)
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
