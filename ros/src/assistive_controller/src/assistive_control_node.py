#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Float32, Bool
from std_msgs.msg import Bool
import random
import time
import numpy as np
import sys, os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(1, str(project_root) +'/')

import config as cfg
cfg = cfg.config()
sys.path.insert(1, cfg.lib_path)
from assistive_control import AssistiveControl

class AssistiveControlNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('AssistiveControlNode')

        self.ass_active = rospy.get_param('/ass_active', 0)


        ## assistive controller settings
        self.cfg = cfg
        A = cfg.A
        B = cfg.B
        Cx = cfg.Cx
        cx = cfg.cx
        G = cfg.Gu
        g = cfg.gu
        Cx_dis_list = cfg.Cx_dis_list
        cx_dis_list = cfg.cx_dis_list


        # define frequency of the node
        self.assistive_sys = AssistiveControl(A, B, CFxmat = Cx, cfxvec = cx, Gumat = G, guvec = g,
                    C_list=Cx_dis_list, c_list=cx_dis_list,
                    M=10e8, u_ref=None,
                    gurobi_params=None, method = "gurobi", warm_start=False)
        

        # Initialize publisher for the state
        self.state_sub = rospy.Subscriber('/state', Float64MultiArray, self.state_callback)
        self.state = None
        # self.state = np.array([0.695, 0.160, 0.055, 0.055])
        
        # Initialize subscriber to /user_input topic
        self.control_sub = rospy.Subscriber('/user_input', Float32MultiArray, self.control_callback)

        # assistive system corrected control input
        self.control_pub = rospy.Publisher('/assistive_input', Float64MultiArray, queue_size=1)
        
        # Create a publisher object that will publish Bool messages for feasibility of the problem
        self.feasible_pub = rospy.Publisher('/problem_feasible', Bool, queue_size=1)
        
        # publish time for assistive system computation time
        self.assistive_freq_pub =  rospy.Publisher('/controller_hz', Float32, queue_size=1)
        self.binary_variable_pub =  rospy.Publisher('/binary_var', Float64MultiArray, queue_size=1)

        self.rate = rospy.Rate(cfg.controller_frequency)  

        # Variables to hold the latest state and control data
        self.control = None
        self.prev_control = None
        self.verbose = True
        self.counter = 0
        self.prev_feasible = 0
        self.ass_active = True

    def state_callback(self, msg):
        """Callback function to handle state updates."""
        self.state = msg.data
        if self.verbose:
            rospy.loginfo(f"Received state: {self.state}")

    def control_callback(self, msg):
        """Callback function to handle user control input updates."""
        self.control = msg.data
        if self.verbose:
            rospy.loginfo(f"Received user control: {self.control}")

    def assistive_control(self):
        u = [0.0, 0.0]

        u_user = self.control
        curr_state = self.state
        x_star = np.array([])
        if  u_user != None and curr_state != None:
            if self.counter == 0: 
                rospy.loginfo("Assistive Control intialized")
                self.assistive_sys.init_prob(np.array(curr_state))
                 
            xref = []
            x0 = np.array(curr_state)
            xg = np.array([])
            t0 = time.perf_counter()
            self.assistive_sys.update_prob(x0, np.array(u_user))
            t1 = time.perf_counter()
            self.assistive_sys.solve_prob()
            feasible = self.assistive_sys.feasible
            freq = 1/(time.perf_counter()-t0)
            pPred = [2]

            if feasible and self.ass_active:
                u_asst = self.assistive_sys.u_asst
                x_next = self.assistive_sys.xnext
                pPred = self.assistive_sys.p_bin
                
                self.prev_control = u_asst
                u = u_asst
            else:
                u = np.array(u_user) ## following user control in case of infeasible but it's dangerous
                x_next = np.array(curr_state)
                # u = [0.0, 0.0]

            if self.verbose :
                rospy.loginfo(f"Received State: {repr(x0.tolist())}, Next step: {repr(x_next.tolist())} , "
                               f"\n user ctrl: {repr(u_user)}, asst ctrl: {repr(u)}, Feasible: {bool(feasible)}"
                               f"Controller Frequency: {freq}")

            # publish controller computation time:
            comp_freq = Float32()
            comp_freq.data = freq
            self.assistive_freq_pub.publish(comp_freq)
            
            # Publish message for the corrected control input [ax, ay]
            control_msg = Float64MultiArray()
            control_msg.data = u.tolist()
 
            self.control_pub.publish(control_msg)

            # publish if the solver is feasible
            bool_msg = Bool()
            bool_msg.data =  bool(feasible)

            # Publish the message
            self.feasible_pub.publish(bool_msg)
            
            # publish binary variables
            binary_var_msg = Float64MultiArray()
            binary_var_msg.data = pPred
            self.binary_variable_pub.publish(binary_var_msg)

            self.counter += 1
            self.prev_feasible = feasible

        else:
            rospy.loginfo(" self.state != None and self.control!=None")

    def spin(self):
        """Keep the node running and processing callbacks."""
        while not rospy.is_shutdown():

            self.assistive_control()
            self.rate.sleep()  # Sleep to maintain the loop at the desired frequency


if __name__ == '__main__':
    try:
        node = AssistiveControlNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
