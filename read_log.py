import pickle

def get_outcomes(logfile):
    data = pickle.load(open(logfile, 'rb'))
    position = data[-1].race_position
    total_distance = data[-1].distance_raced
    old_laptime = 1
    for i in data:
        if i.last_lap_time == 0 and old_laptime !=1:

        er

get_outcomes('/home/mark/Documents/AI/CI/Torcs_JH_PW_MR/drivelogs/drivelog-2017-11-21-15-44-48.pickle')
