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
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    lr = pickle.load(open('data2.pkl', 'rb'))
    prediction = lr.predict(final)[0]
    return render_template('index.html', prediction=prediction)
if __name__ == '__main__':
    app.run()
