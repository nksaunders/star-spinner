from flask import Flask, request, render_template, jsonify, Response

app = Flask(__name__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import load_model
import theano.tensor as T

import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@app.route('/')
def my_form():
    return render_template('index.html')

# @app.route('/', methods=['POST'])
# def my_form_post():
#     text = request.form['text']
#     processed_text = text.upper()
#     return processed_text

@app.route('/join', methods=['POST', 'GET'])
def infer_stellar_params(mlt=1.9, fk=7.2, rocrit=2.2):

    path = 'data/models/model_4'
    model = load_model(path)

    age = float(request.form['age'])
    feh = float(request.form['feh'])
    mass = float(request.form['mass'])

    solar_test_inputs = T.stack([np.log10(age), mass, feh, mlt, fk, rocrit])

    weights = model.get_weights()

    input_offset = model.layers[0].mean.numpy()
    input_scale = np.sqrt(model.layers[0].variance.numpy())

    output_offset = np.array(model.layers[-1].offset)
    output_scale = np.array(model.layers[-1].scale)

    w = weights[3:-1:2]  # hidden layer weights
    b = weights[4::2]    # hidden layer biases

    def emulate(inputs):
        inputs = (inputs - input_offset) / input_scale
        for wi, bi in zip(w[:-1], b[:-1]):
            inputs = T.nnet.elu(T.dot(inputs, wi) + bi)
        outputs = T.dot(inputs, w[-1]) + b[-1]
        outputs = output_offset + outputs * output_scale
        return outputs

    outputs = emulate(solar_test_inputs)

    y = outputs.eval()

    output_dict = {'Teff':f'{10**y[0]:.2f} K',
                   'R': f'{10**y[1]:.2f} solar radii',
                   'Z': f'{y[2]:.2f}',
                   'Prot': f'{10**y[3]:.2f} days'}

    return jsonify(result=output_dict)

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

if __name__ == "__main__":
    app.run(debug=True)