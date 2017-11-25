import logging

import math

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController

import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

sensors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# sensors = [0, 3, 6, 8, 10, 12, 15, 18]



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

class Driver:
    """
    Driving logic.

    Implement the driving intelligence in this class by processing the current
    car state as inputs creating car control commands as a response. The
    ``drive`` function is called periodically every 20ms and must return a
    command within 10ms wall time.
    """



    def __init__(self, logdata=True):
        self.steering_ctrl = CompositeController(
            ProportionalController(0.1),
            IntegrationController(0.1, integral_limit=1.5),
            DerivativeController(0.5)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None

        self.NN = Net()
        self.NN.load_state_dict(torch.load('nn_5'))

    

    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].

        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return -90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
            30, 45, 60, 75, 90

    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """        
        distances = np.array(carstate.distances_from_edge)
        distances = distances[sensors]

        inp = np.append(distances, [carstate.angle, carstate.speed_x, carstate.distance_from_center])

        inp = torch.from_numpy(inp).type(torch.FloatTensor)
        inp = Variable(inp)

        out = self.NN.forward(inp)

        command = Command()
        # command.accelerator = out.data[0]
        if out.data[0] > 0.2:
            command.accelerator = 1
        else:
            command.accelerator = 0.5
        
        command.brake = out.data[1]

        command.steering = out.data[2]
        if distances[0] == -1:
            if carstate.speed_x >= 10:
                command.accelerator = 0
                command.brake = 0.4
            else:
                command.accelerator = 0.8
                command.brake = 0

            max_angle = 10
            if carstate.distance_from_center >= 4.0:
                max_angle = 80

            if carstate.distance_from_center >= 0:
                if carstate.angle <= max_angle:
                    command.steering = -2 / (carstate.speed_x + 1)
                elif carstate.angle >= (max_angle + 5):
                    command.steering = 2 / (carstate.speed_x + 1)
                else:
                    command.steering = 0
            else:
                if carstate.angle >= -max_angle:
                    command.steering = 2 / (carstate.speed_x + 1)
                elif carstate.angle <= -(max_angle + 5):
                    command.steering = -2 / (carstate.speed_x + 1)
                else:
                    command.steering = 0

        print("\033c")
        print("sensors: \n", distances, "\n \nangle:", carstate.angle, "\nspeed:", carstate.speed_x, "\ndist from cent:", carstate.distance_from_center, "\n")
        print("acc", out.data[0], "\nbrake", out.data[1], "\nsteer", out.data[2])

        if carstate.speed_x <= 10:
            command.accelerator = 0.8

        ACC_LATERAL_MAX = 6400 * 5


        self.accelerate(carstate, 0.0, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        # speed_error = 1.0025 * target_speed - carstate.speed_x
        # acceleration = self.acceleration_ctrl.control(
        #     speed_error,
        #     carstate.current_lap_time
        # )

        # # stabilize use of gas and brake:
        # acceleration = math.pow(acceleration, 3)

        # if acceleration > 0:
        #     if abs(carstate.distance_from_center) >= 1:
        #         # off track, reduced grip:
        #         acceleration = min(0.4, acceleration)

        #     command.accelerator = min(acceleration, 1)

        #     if carstate.rpm > 8000:
        #         command.gear = carstate.gear + 1

        # else:
        #     command.brake = min(-acceleration, 1)

        # command.accelerator = 0
        # command.brake = 0 
        # if (target_speed - carstate.speed_x) > 20:
        #     command.accelerator = 1
        # elif (target_speed - carstate.speed_x) > 0:
        #     command.accelerator = (target_speed - carstate.speed_x) / 20
        # elif (target_speed - carstate.speed_x) > -20:
        #     command.brake = -(target_speed - carstate.speed_x) / 20
        # else:
        #     command.brake = 1

        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, carstate, target_track_pos, command):


        steering_error = target_track_pos - carstate.distance_from_center
        desired_angle = steering_error * -0.5
        angle_error = desired_angle - carstate.angle
        command.steering = -angle_error * 0.05
        # command.steering = self.steering_ctrl.control(
        #     steering_error,
        #     carstate.current_lap_time
        # )
