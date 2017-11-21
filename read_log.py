import pickle
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

files = []

for file in os.listdir(os.path.join(os.getcwd() + '/drivelogs')):
	if file.endswith(".pickle"):
		files.append(os.path.join(os.path.join(os.getcwd() + '/drivelogs'), file))

sensors = [0, 5, 8, 10, 11, 13, 18]

all_states = []
for filepath in files:
	with open(filepath, 'rb') as logfile:
		unpickler = pickle.Unpickler(logfile)
		try:
			while True:
				state, command = unpickler.load()
				all_states.append(state)
		except EOFError:
			pass

inputs = np.zeros((len(all_states), len(sensors)))
outputs = np.zeros((len(all_states), 2))
for index, state in enumerate(all_states):
	distances = np.array(state.distances_from_edge)
	distances = distances[sensors]
	outputs[index, :] = [state.speed_x, state.distance_from_center]
	inputs[index, :] = list(distances)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(7, 1000)
        # self.bias1 = nn.Linear()
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(1000, 2)

    def forward(self, x):
    	x = self.layer1(x)
    	x = self.relu(x)
    	x = self.layer2(x)
    	return x


net = Net()

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.00000003, momentum=0.0)
# create a loss function
criterion = nn.MSELoss()

epochs = 100

# run the main training loop
for epoch in range(epochs):
	for index in range(len(inputs)):
		data = torch.from_numpy(inputs[index, :]).type(torch.FloatTensor)
		target = torch.from_numpy(outputs[index, :]).type(torch.FloatTensor)
		
		data, target = Variable(data), Variable(target)
		net.zero_grad()
		net_out = net(data)

		loss = criterion(net_out, target)
		loss.backward()

		optimizer.step()
	print("Train Epoch", epoch, "loss", loss.data[0])
	torch.save(net.state_dict(), "nn_3")

# pickle.dump(net, open( "nn_1.p", "wb" ) )
