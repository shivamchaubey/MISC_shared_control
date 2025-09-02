#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray
import random
import numpy as np

import sys, os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root) +'/')
import config as cfg
cfg = cfg.config()


class SystemSimulationNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('SystemSimulationNode')

        # Initialize publisher for the state
        self.state_pub = rospy.Publisher('/state', Float64MultiArray, queue_size=1)
        
        # Initialize subscriber for assistive input
        self.control_sub = rospy.Subscriber('/assistive_input', Float64MultiArray, self.control_callback)
        self.restart_sim = rospy.get_param('/restart_sim', True)

        self.A = cfg.A
        self.B = cfg.B

        self.curr_x = cfg.spawn_x0

        # Initialize state and control input variables
        self.init_x = self.curr_x.copy()
        self.u = np.array([0.0, 0.0])  # Default control input
        self.nx = self.curr_x.shape[0]
        self.apply_disturbance = cfg.disturbance
       
        if self.apply_disturbance:
            self.E = cfg.E
            self.Gw = cfg.Gw
            self.Fw = cfg.Fw
            self.rng = np.random.default_rng(10)

        # Set up the rate for publishing 
        self.rate = rospy.Rate(cfg.simulation_frequency) 


    def control_callback(self, msg):
        """Callback function to handle control input updates."""
        # Extract control input from the received message
        self.u = np.array(msg.data)  # Update control input with received data
        # rospy.loginfo(f"Received control input: {self.u}")
    
    def publish_state_and_control(self):
        """Publish the current state and altered control input."""
        # Create a message for the state [x, y, vx, vy]

        curr_x = self.curr_x
        u = self.u

        # Simulate the system's next state using the current state and control input
        next_state = self.A@curr_x + self.B@u
        if self.apply_disturbance:

            nw = self.E.shape[1]
            I  = np.eye(nw)
            assert self.Gw.shape == (2*nw, nw), "Expect Gw = [I; -I] for simple box."
            assert np.allclose(self.Gw[:nw], I) and np.allclose(self.Gw[nw:], -I)

            ub = self.Fw[:nw].reshape(nw,1)      # upper bounds
            lb = -self.Fw[nw:].reshape(nw,1)     # lower bounds
            w  = np.random.uniform(lb, ub, size=(nw,1))

            # w = lb + (ub - lb) * np.random.rand(nw, 1)
            next_state += self.E @ (w.squeeze())

        # Prepare the state message
        state_msg = Float64MultiArray()
        state_msg.data = next_state.tolist()
        # rospy.loginfo(f"SIM >> curr state: {curr_x}, next state: {state_msg.data}, control: {u}")
        # Publish the state message
        self.state_pub.publish(state_msg)
        
        # Update the current state
        self.curr_x = next_state
        # self.next_state = next_state
        
        
    def spin(self):
        """Keep the node running and processing callbacks."""
        while not rospy.is_shutdown():
            # Continuously publish the state at a set frequency
            self.publish_state_and_control()
            self.rate.sleep()  # Sleep to maintain the loop at the desired frequency

if __name__ == '__main__':
    try:
        # Create the ROS node object and run it
        node = SystemSimulationNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass


