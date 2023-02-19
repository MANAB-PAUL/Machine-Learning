import numpy as np
import math as math
import nnfs


e = math.e
nnfs.init()
# ACTIVATION FUNCTIONS FOR BATCH INPUTS
layer_output = [
    [4.8, 1.21, 2.385],  # output input set 1 output[input1][neuron1], output[inp2][n2]...
    [8.9, -1.81, 0.2],  # outputs for input set 2
    [1.41, 1.051, 0.026]  # outputs for input set 3
]

###   SOFTMAX    FUNCTION   ###

def softmax(layer_output):
# exponentiation
    exp_values = []
    for outputs in layer_output:
        exp_rows = []
        maxx = max(outputs)
        for output in outputs:
            exp_rows.append(e ** (output - maxx))
        exp_values.append(exp_rows)


    ###    NORMALIZING   ###
    norm_values = []
    for exp_row in exp_values:
        denominator = sum(exp_row)
        norm_row = []
        for numerator in exp_row:
            norm_row.append(numerator / denominator)
        norm_values.append(norm_row)
    return norm_values

# testing the softmax function
for row in softmax(layer_output):
    print(row)
    print(sum(row))



### rectified linear function
def rectified_liner_function(layer_outputs):
    recl = []
    for layer_output in layer_outputs:
        recl_row = []
        for output in layer_output:
            recl_row.append(max(output, 0))
        recl.append(recl_row)
    return recl

### step function
def step_function(layer_outputs):
    step_outputs = []
    for layer_output in layer_outputs:
        step_output = []
        for output in layer_output:
            if(output > 0):
                step_output.append(1)
            else:
                step_output.append(0)
        step_outputs.append(step_output)
    return step_outputs



### signoid function
def signoid_function(layer_outputs):
    recl = []
    for layer_output in layer_outputs:
        recl_row = []
        for output in layer_output:
            recl_row.append(1 / (1 + (e ** output)))
        recl.append(recl_row)
    return recl
