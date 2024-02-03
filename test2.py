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
from build_neural_networks import BuildActorNetwork

test_IRRELEVANT_STATES = [0,1,2,8,9,10,11,12]

import matplotlib
matplotlib.use('Agg') # This removes Runtime ("RuntimeError: main thread is not in main loop") and User Errors ("UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.")
import matplotlib.pyplot as plt

environment_file = __import__('environment_' + Settings.ENVIRONMENT)                

"""
*# Relative pose expressed in the chaser's body frame; everything else in Inertial frame #*
Deep guidance output in x and y are in the chaser body frame
"""


CHECK_VELOCITY_LIMITS_IN_PYTHON = True


def make_C_bI(angle):        
    C_bI = np.array([[ np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]]) # [2, 2]        
    return C_bI
 

class DeepGuidanceModelRunner:
    
    def __init__(self):
        
        print("Initializing deep guidance model runner")
        self.is_running = True

        # Initialize an environment so we can use its methods
        self.environment = environment_file.Environment()
        self.environment.reset(True,True)

        # Initialize states of spacecraft
        self.chaser_position   = self.environment.state
        self.chaser_velocity   = np.array([0.,0.,0.])
        self.target_position   = self.environment.target_location
        self.target_velocity   = np.array([0.,0.,self.environment.TARGET_ANGULAR_VELOCITY])
        self.obstac_position   = self.environment.obstacle_location
        self.obstac_velocity   = self.environment.OBSTABLE_VELOCITY
        
        # Clear any old graph
        tf.reset_default_graph()
        
        # Initialize Tensorflow, and load in policy
        self.sess = tf.Session()
        # Building the policy network
        self.state_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, Settings.OBSERVATION_SIZE], name = "state_placeholder")
        self.actor = BuildActorNetwork(self.state_placeholder, scope='learner_actor_main')
    
        # Loading in trained network weights
        print("Attempting to load in previously-trained model\n")
        saver = tf.train.Saver() # initialize the tensorflow Saver()
    
        # Try to load in policy network parameters
        try:
            filelocation = r"C:\Users\court\Desktop\Kirk_DRL\Tensorboard\Current\Kirkados_default_run-2024-01-23_14-50"
            ckpt = tf.train.get_checkpoint_state(filelocation)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("\nModel successfully loaded!\n")
    
        except (ValueError, AttributeError):
            print(ckpt)
            # print("Model: ", ckpt.model_checkpoint_path, " not found... :(")
            raise SystemExit
        
        print("Done initializing model!")

    def run(self):

        print("Running Deep Guidance!")
        
        counter = 1
        # Parameters for normalizing the input
        relevant_state_mean = np.delete(Settings.STATE_MEAN, Settings.IRRELEVANT_STATES)
        relevant_half_range = np.delete(Settings.STATE_HALF_RANGE, Settings.IRRELEVANT_STATES)
        
        # To log data
        deep_guidance_log = []
        total_state_log = []
        cumulative_reward_log = [0]
        
        # Run zeros through the policy to ensure all libraries are properly loaded in
        deep_guidance = self.sess.run(self.actor.action_scaled, feed_dict={self.state_placeholder:np.zeros([1, Settings.OBSERVATION_SIZE])})[0]            

        ####################
        ### Getting Data ###
        #################### 

        # Get the total state from the environment:
        #   [x, y, theta, \
        #   desired_x_error, desired_y_error, desired_theta_error, \
        #   obstable_distance_x, obstacle_distance_y, \
        #   target_location_x, target_location_y, target_location_theta, \
        #   obstacle_location_x, obstacle_location_y] 
        total_state = self.environment.make_total_state() 
        total_state_log.append(total_state)

        while self.is_running:


            
            


            # self.environment.chaser_position   = np.array([Pi_red_x, Pi_red_y, Pi_red_theta])
            # self.environment.chaser_velocity   = np.array([Pi_red_Vx, Pi_red_Vy, Pi_red_omega])
            # self.environment.target_position   = np.array([Pi_black_x, Pi_black_y, Pi_black_theta])
            # self.environment.target_velocity   = np.array([Pi_black_Vx, Pi_black_Vy, Pi_black_omega])

            #################################
            ### Building the Policy Input ###
            ################################# 
            policy_input = np.delete(total_state, test_IRRELEVANT_STATES)

            # Normalizing            
            if Settings.NORMALIZE_STATE:
                normalized_policy_input = (policy_input - relevant_state_mean)/relevant_half_range
            else:
                normalized_policy_input = policy_input

            # Reshaping the input    
            normalized_policy_input = normalized_policy_input.reshape([-1, Settings.OBSERVATION_SIZE])

            # Run processed state through the policy
            # [v_x, v_y, omega]
            deep_guidance = self.sess.run(self.actor.action_scaled, feed_dict={self.state_placeholder:normalized_policy_input})[0] # [v_x, v_y, omega]

            # Rotating the command into the inertial frame
            # if not Settings.ACTIONS_IN_INERTIAL:
            #     deep_guidance[0:2] = np.matmul(make_C_bI(Pi_red_theta).T,deep_guidance[0:2])

            # if CHECK_VELOCITY_LIMITS_IN_PYTHON:                    
            #     current_velocity = self.environment.state[self.environment.POSITION_STATE_LENGTH:]             
            #     deep_guidance[:len(current_velocity)][(np.abs(current_velocity) > Settings.VELOCITY_LIMIT[:len(current_velocity)]) & (np.sign(deep_guidance[:len(current_velocity)]) == np.sign(current_velocity))] = 0

            # print(deep_guidance)

            ####################################
            ### Stepping Through Environment ###
            ####################################
            reward = self.environment.step(deep_guidance)[1] # return reward only

            ####################
            ### Getting Data ###
            #################### 
            #   [x, y, theta, \
            #   desired_x_error, desired_y_error, desired_theta_error, \
            #   obstable_distance_x, obstacle_distance_y, \
            #   target_location_x, target_location_y, target_location_theta, \
            #   obstacle_location_x, obstacle_location_y]  
            total_state = self.environment.make_total_state() 
            
            ####################
            ### Data Logging ###
            #################### 
            total_state_log.append(total_state)
            deep_guidance_log.append(deep_guidance)
            cumulative_reward_log.append(cumulative_reward_log[-1] + reward)


            # data_log.append([deep_guidance[0], deep_guidance[1], deep_guidance[2], \
            #                      Pi_red_x, Pi_red_y, Pi_red_theta, \
            #                      Pi_red_Vx, Pi_red_Vy, Pi_red_omega,        \
            #                      Pi_black_x, Pi_black_y, Pi_black_theta,    \
            #                      Pi_black_Vx, Pi_black_Vy, Pi_black_omega])

            if self.environment.is_done():
                break

        # Render an animation of the test
        environment_file.render(np.asarray(total_state_log), [], total_state_log[0][8:10+1], [], np.asarray(cumulative_reward_log), [], [], [], [], [], [], 0, 'test_run', "C:/Users/court/Desktop/")

        # print(total_state_log)

        plt.figure()
        plt.plot(np.transpose(total_state_log)[0],np.transpose(total_state_log)[1],'r-')
        plt.plot(np.transpose(total_state_log)[8],np.transpose(total_state_log)[9],'ko')
        plt.plot(np.transpose(total_state_log)[11],np.transpose(total_state_log)[12],'bo')
        plt.savefig("foo_xy.png")
        plt.figure()
        plt.plot(np.transpose(total_state_log)[2])
        plt.savefig("foo_theta.png")

        # Close tensorflow session
        self.sess.close()


# Initialize Deep Guidance Model
deep_guidance_model = DeepGuidanceModelRunner()
 
deep_guidance_model.run()

print('Done :)')