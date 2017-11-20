import pickle

filepath = "drivelogs/forza.pickle"
all_states = []
with open(filepath, 'rb') as logfile:
	unpickler = pickle.Unpickler(logfile)
	try:
		while True:
			state, command = unpickler.load()
			all_states.append(state)
	except EOFError:
		pass

# print(len(all_states))
print(type(all_states[1]))