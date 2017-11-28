import pickle

def get_outcomes(logfile):
	unpickler = pickle.Unpickler(open(logfile, 'rb'))
	state, command = unpickler.load()
	all_states = []
	try:
	    while True:
	    	state, command = unpickler.load()
	    	all_states.append(state)
	except EOFError:
		pass

	position = all_states[-1].race_position
	laptimes = []
	for entry in all_states:
		if entry.last_lap_time != 0:
			laptimes.append(entry.last_lap_time)
	unique_laptimes = set(laptimes)
	return position, unique_laptimes

pos, times = get_outcomes('drivelogs/forza.pickle')
print(pos, times)