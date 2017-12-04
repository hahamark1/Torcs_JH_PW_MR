import pickle
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

files = []

for file in os.listdir(os.path.join(os.getcwd() + '/drive_data')):
	if file.endswith(".pickle"):
		files.append(os.path.join(os.path.join(os.getcwd() + '/drive_data'), file))

sensors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

all_states = []
for filepath in files:
	with open(filepath, 'rb') as logfile:
		# print(logfile)
		unpickler = pickle.Unpickler(logfile)
		try:
			while True:
				state, command = unpickler.load()
				all_states.append(state)
		except EOFError:
			pass

inputs = np.zeros((len(all_states), len(sensors) + 3))
outputs = np.zeros((len(all_states), 3))
for index, state in enumerate(all_states):
	distances = np.array(state.distances_from_edge)
	distances = distances[sensors]
	outputs[index, :] = [state.accel_cmd, state.brake_cmd, state.steer_cmd]
	inputs[index, :] = list(distances) + [state.angle, state.speed_x, state.distance_from_center]


print('n entries:', len(outputs))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(len(sensors) + 3, 1000)
        self.layer2 = nn.Linear(1000, 100)
        self.layer3 = nn.Linear(100, 3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        x = self.tanh(x)
        x = self.layer3(x)
        x = self.tanh(x)
        return x


net = Net().cuda()

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.0)
# create a loss function
criterion = nn.MSELoss()

epochs = 10000

# run the main training loop
for epoch in range(epochs):
	for index in range(0,len(inputs),25):
		data = torch.from_numpy(inputs[index: index+25, :]).type(torch.FloatTensor).cuda()
		target = torch.from_numpy(outputs[index: index+25, :]).type(torch.FloatTensor).cuda()

		data, target = Variable(data), Variable(target)
		net.zero_grad()
		net_out = net(data)

		loss = criterion(net_out, target)
		loss.backward()

		optimizer.step()
	print("Train Epoch", epoch, "loss", loss.data[0])
	torch.save(net.state_dict(), "nn_6")
