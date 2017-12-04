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

for i in all_states[-100:]:
    print(i)
