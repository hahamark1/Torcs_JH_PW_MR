import pickle
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# def read_data():
files = []

for file in os.listdir(os.path.join(os.getcwd() + '/drivelogs')):
	if file.endswith(".pickle"):
		files.append(os.path.join(os.path.join(os.getcwd() + '/drivelogs'), file))

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

inputs = np.zeros((len(all_states), len(all_states[1].distances_from_edge) + 1))
outputs = np.zeros((len(all_states), 2))
for index, state in enumerate(all_states):
	outputs[index, :] = [state.speed_x, state.distance_from_center]
	inputs[index, :] = list(state.distances_from_edge) + [state.angle]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(20, 50)
        self.layer2 = nn.Linear(50, 2)

    def forward(self, x):
    	print(x)
    	x = self.layer1(x)
    	print(x)
    	x = self.layer2(x)
    	return x

net = Net()
# input1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
input1 = torch.zeros(20)
net_out = net(input1)

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# create a loss function
criterion = nn.NLLLoss()

epochs = 20

# run the main training loop
for epoch in range(epochs):
	for index in range(len(inputs)):
		data = torch.from_numpy(inputs[index, :])
		target = torch.from_numpy(outputs[index, :])
		print(type(data), type(target))
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		net_out = net(data)
		print(type(net_out))
		loss = criterion(net_out, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
		    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		            epoch, batch_idx * len(data), len(train_loader.dataset),
		                   100. * batch_idx / len(train_loader), loss.data[0]))