import pickle
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

files = []

for file in os.listdir(os.path.join(os.getcwd() + '/train_data')):
	if file.endswith(".pickle"):
		files.append(os.path.join(os.path.join(os.getcwd() + '/train_data'), file))

sensors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

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
		self.layer1 = nn.Linear(len(sensors) + 3 + 1000, 1000)
		self.layer2 = nn.Linear(1000, 1000)
		# self.layer3 = nn.Linear(100, 100)
		# self.layer4 = nn.Linear(100, 100)
		self.layer3 = nn.Linear(1000, 3)
		self.tanh = nn.Tanh()

	def forward(self, x, hidden):
		combined = torch.cat((x, hidden), 0)
		hidden = self.layer1(combined)
		x = self.tanh(hidden)
		x = self.layer2(x)
		x = self.tanh(x)
		x = self.layer3(x)
		output = self.tanh(x)      
		# x = self.layer4(x)
		# x = self.tanh(x) 
		# x = self.layer5(x)
		# x = self.tanh(x) 
		return output, hidden


net = Net()

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.0)
# create a loss function
criterion = nn.MSELoss()

epochs = 10000
hidden = Variable(torch.zeros(1000))

# run the main training loop
for epoch in range(epochs):

	for index in range(len(inputs)):
		if index % 100 == 0:
			hidden = Variable(torch.zeros(100))
		data = torch.from_numpy(inputs[index, :]).type(torch.FloatTensor)
		target = torch.from_numpy(outputs[index, :]).type(torch.FloatTensor)
		
		data, target = Variable(data), Variable(target)
		net.zero_grad()
		net_out, hidden = net(data, hidden)


		loss = criterion(net_out, target)
		loss.backward(retain_graph=True)

		optimizer.step()
		if index % 100 == 0:
			print("Progress: ", index/len(inputs)*100, "%")
	print("Train Epoch", epoch+1, "loss", loss.data[0])
	torch.save(net.state_dict(), "rnn_1")