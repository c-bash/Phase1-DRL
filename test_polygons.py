"""
This script loads in a trained policy neural network and uses it for inference.

Typically this script will be executed on the Nvidia Jetson TX2 board during an
experiment in the Spacecraft Robotics and Control Laboratory at Carleton
University.

Script created: June 12, 2019
@author: Kirk (khovell@gmail.com)
"""

import tensorflow as tf
import numpy as np
import time

from settings import Settings
# from build_neural_networks import BuildActorNetwork

# test_IRRELEVANT_STATES = [0,1,2,8,9,10,11,12]

import matplotlib
matplotlib.use('Agg') # This removes Runtime ("RuntimeError: main thread is not in main loop") and User Errors ("UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.")
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon # for collision detection

environment_file = __import__('environment_' + Settings.ENVIRONMENT)                

"""
*# Relative pose expressed in the chaser's body frame; everything else in Inertial frame #*
Deep guidance output in x and y are in the chaser body frame
"""

# Initialize an environment so we can use its methods
env = environment_file.Environment()
env.reset(True,True)

polygons = env.check_collisions()

tar_poly = polygons[0]
for_poly = polygons[1]
sol_poly = polygons[2]
obs_poly = polygons[3]
cha_poly = polygons[4]
ee_point = polygons[5]



# print(tar_poly)
# print(for_poly)
# print(cha_poly)
# print(ee_point)

plt.figure()
plt.plot(tar_poly.exterior.xy[0],tar_poly.exterior.xy[1],'k')
plt.plot(for_poly.exterior.xy[0],for_poly.exterior.xy[1],'b')
plt.plot(sol_poly.exterior.xy[0],sol_poly.exterior.xy[1],'b')
plt.plot(obs_poly.exterior.xy[0],obs_poly.exterior.xy[1],'b')
plt.plot(cha_poly.exterior.xy[0],cha_poly.exterior.xy[1],'r')
plt.scatter(ee_point.x,ee_point.y)
# plt.plot(np.transpose(total_state_log)[8],np.transpose(total_state_log)[9],'ko')
# plt.plot(np.transpose(total_state_log)[11],np.transpose(total_state_log)[12],'bo')
plt.xlim([0,3.5])
plt.ylim([0,2.4])
plt.savefig("foo_polygons.png")


print('Done :)')