import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_3.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('cynthia.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output>2:
        return render_template('cynthia.html',prediction_text="The tumor is malignant which means you have breast cancer".format(output))
    else:
        return render_template('cynthia.html', prediction_text='The tumor is benign which means you dont have breast cancer'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
