import tensorflow as tf
import numpy as np

INPUT_NEURONS = 22
HIDDEN_LAYER_1_NEURONS = 500
OUTPUT_NEURONS = 3

import csv
import numpy as np
from scipy.special import expit
output = []
input = []

with open('aalborg.csv', 'r') as csv_file:
    aalborg_data = csv.reader(csv_file, delimiter=',', quotechar='|')
    for datapoint in aalborg_data:
        if datapoint[0] !='ACCELERATION':
            output.append(datapoint[:3])
            input.append(datapoint[3:])

with open('f-speedway.csv', 'r') as csv_file:
    speedway_data = csv.reader(csv_file, delimiter=',', quotechar='|')
    for datapoint in speedway_data:
        if datapoint[0] !='ACCELERATION':
            output.append(datapoint[:3])
            input.append(datapoint[3:])

with open('alpine-1.csv', 'r') as csv_file:
    alpine_data = csv.reader(csv_file, delimiter=',', quotechar='|')
    for datapoint in alpine_data:
        if datapoint[0] !='ACCELERATION':
            output.append(datapoint[:3])
            input.append(datapoint[3:])

train_in, train_out = [], []
for i in range(len(input)):
    if len(input[i]) == 22:
        train_in.append(input[i])
        train_out.append(output[i])
        # output.pop(i)
        # print(len(input[2]))
train_ctrl,train_output = np.array(train_in,dtype=float), np.array(train_out,dtype=float)
# test_ctrl, test_output = np.array(train_in[20000:],dtype=float), np.array(train_out[20000:],dtype=float)

x = train_ctrl
y = train_output

h1_layer = {'weight': tf.Variable(np.random.rand(INPUT_NEURONS, HIDDEN_LAYER_1_NEURONS).astype(np.float32)),
            'bias': tf.constant(0.0)}
output_layer = {'weight': tf.Variable(np.random.rand(HIDDEN_LAYER_1_NEURONS, OUTPUT_NEURONS).astype(np.float32)),
                'bias': tf.constant(0.0)}


h1 = tf.matmul(x, h1_layer['weight']) + h1_layer['bias']
h1 = tf.sigmoid(h1)

predict = tf.matmul(h1, output_layer['weight']) + output_layer['bias']

loss = tf.reduce_mean(tf.square(predict - y))
optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
#
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(50000):
		if i % 5000 == 0:
			print('Loss: ', sess.run(loss, {x: x_data, y: y_data}))
		sess.run(optimizer, {x: x_data, y: y_data})

# print sess.run(predict, {x: [[1, 1]] })
