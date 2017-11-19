import pickle

filepath = "drivelogs/drivelog-2017-11-19-22-03-49.pickle"
all_states = []
with open(filepath, 'rb') as logfile:
	unpickler = pickle.Unpickler(logfile)
	try:
		while True:
			state, command = unpickler.load()
			all_states.append(state)
	except EOFError:
		pass

print(len(all_states))