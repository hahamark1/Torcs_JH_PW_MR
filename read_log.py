import pickle
import numpy as np
import os

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