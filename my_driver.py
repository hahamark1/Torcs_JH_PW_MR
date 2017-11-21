from pytocl.driver import Driver
from pytocl.car import State, Command
import pyESN
import pickle
import numpy as np



class MyDriver(Driver):

    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State):
    #     # Interesting stuff
    #     command = Command()
    #     speed = (carstate.speed_x + carstate.speed_y)/2

    #     input = np.append([speed, carstate.distance_from_center, carstate.angle],[x for x in carstate.distances_from_edge])
    #     output = NN.predict(np.array([input]))[0]
    #     print(output)
    #     self.steer = output[2]
    #     self.accelerate = output[0]
    #     self.brake = output[1]
    #     return command
