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
train_ctrl,train_output = np.array(train_in[:20000],dtype=float), np.array(train_out[:20000],dtype=float)
test_ctrl, test_output = np.array(train_in[20000:],dtype=float), np.array(train_out[20000:],dtype=float)


import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN
import pickle

rng = np.random.RandomState(42)

esn = ESN(n_inputs = 22,
          n_outputs = 3,
          n_reservoir = 200,
          spectral_radius = 0.25,
          sparsity = 0.95,
          noise = 0.001,
          input_shift = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          input_scaling = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
          teacher_scaling = 1.12,
          teacher_shift = -0.7,
          out_activation = np.tanh,
          inverse_out_activation = expit,
          random_state = rng,
          silent = False)

pred_train = esn.fit(train_ctrl,train_output)

pickle.dump(esn, open("esn.p", "wb"))


print("test error:")
pred_test = esn.predict(test_ctrl)
print(test_ctrl.shape)
print(np.sqrt(np.mean((pred_test - test_output)**2)))
