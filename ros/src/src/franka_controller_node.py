#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray
import random
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from franky import Robot, Affine, CartesianMotion, CartesianVelocityMotion, \
                   ReferenceType, Twist as FrankyTwist

import sys, os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root) +'/')

import config as cfg
cfg = cfg.config()
sys.path.insert(1, cfg.lib_path)


class SystemSimulationNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('FrankaNode')
        self.robot_ip     = rospy.get_param("~robot_ip", "172.16.0.2")
        self.rel_dyn      = float(rospy.get_param("~relative_dynamics_factor", 0.006))
        self.frame_id     = rospy.get_param("~frame_id", "panda_link0")
        self.motion_type = rospy.get_param("~motion_type", "position").lower()  
        
        # --- robot ---
        rospy.loginfo("Connecting to Franka at %s ...", self.robot_ip)
        self.robot = Robot(self.robot_ip)

        rospy.loginfo("Connected to %s ...", self.robot_ip)
        
        try: self.robot.recover_from_errors()
        except Exception as e: rospy.logwarn("recover_from_errors(): %s", e)
        self.robot.relative_dynamics_factor = self.rel_dyn

        cs = self.robot.current_cartesian_state
        ee_pose = cs.pose.end_effector_pose

        # Use franky.Affine fields
        self.xy0 = np.array([ee_pose.translation[0], ee_pose.translation[1]])
        self.curr_x = np.array([ee_pose.translation[0], ee_pose.translation[1], 0, 0])
        self.prev_x = np.array([ee_pose.translation[0], ee_pose.translation[1], 0, 0])

        print ("self.xy0 ", self.xy0 )
        self.z_pos = 0
        self.z0  = float(ee_pose.translation[2])
        self.q0  = [ee_pose.quaternion[0], ee_pose.quaternion[1],
                    ee_pose.quaternion[2], ee_pose.quaternion[3]]
        

        # Initialize publisher for the state
        self.state_pub = rospy.Publisher('/state', Float64MultiArray, queue_size=1)
        self.curr_state_pub = rospy.Publisher('/curr_state', Float64MultiArray, queue_size=1)
        self.pub_cmd_arr   = rospy.Publisher("/panda_cmd",      Float64MultiArray, queue_size=1)
        self.pub_pose      = rospy.Publisher("franka/ee_pose",  PoseStamped,       queue_size=1)
        self.pub_twist     = rospy.Publisher("franka/ee_twist", TwistStamped,      queue_size=1)
        self.pub_cmd_arr   = rospy.Publisher("/asst_cmd",      Float64MultiArray, queue_size=1)
        # self.pub_cmd_twist = rospy.Publisher("cmd/twist",       TwistStamped,      queue_size=10)

        # Initialize subscriber for assistive input
        self.control_sub = rospy.Subscriber('/assistive_input', Float64MultiArray, self.control_callback)
        self.restart_sim = rospy.get_param('/restart_sim', True)
        self.restart_init_x = rospy.get_param('/init_x', [0, 0, 0, 0])

        # if self.restart_sim == True:
        #     self.curr_x = self.restart_init_x
        # else: 
        #     self.curr_x = np.array([0.695, 0.160, 0.0, 0.0])
        # Define system dynamics
        # Initialize state variables [x, y, vx, vy]
        # self.curr_x = self.restart_init_x 
        self.init_x = self.curr_x.copy()
        # Initialize control input variables
        self.u = np.array([0.0, 0.0])  # Default control input

        # Set up the rate for publishing (e.g., 10 Hz)
        self.rate = rospy.Rate(cfg.simulation_frequency)  # 10 Hz (0.1 seconds)
        # self.rate = rospy.Rate(100)  # 10 Hz (0.1 seconds)
        # self.sys = SystemDynamics(dt = 1/cfg.simulation_frequency, damping_factor = cfg.damping_factor)
        self.A = cfg.A
        self.B = cfg.B
        self.E = cfg.E
        self.Gw = cfg.Gw
        self.gw = cfg.gw

    def control_callback(self, msg):
        """Callback function to handle control input updates."""
        # Extract control input from the received message
        self.u = np.array(msg.data)  # Update control input with received data
        # rospy.loginfo(f"Received control input: {self.u}")
    
    def publish_state_and_control(self):
        """Publish the current state and altered control input."""
        # Create a message for the state [x, y, vx, vy]
        self.restart_sim = rospy.get_param('/restart_sim', False)
        self.restart_init_x = rospy.get_param('/init_x', [0, 0, 0, 0])

        if bool(self.restart_sim) == True:
            self.curr_x = self.restart_init_x 

        curr_x = self.curr_x
        u = self.u
        
        # Simulate the system's next state using the current state and control input
        next_state = self.A@curr_x + self.B@u
        vx, vy = float(next_state[2]), float(next_state[3])
        
        if self.motion_type == "velocity":            
            self.robot.move(CartesianVelocityMotion(FrankyTwist([vx, vy, 0.0], [0.0,0.0,0.0])),
                                asynchronous=True)

        else:        
            self.robot.move(CartesianMotion(Affine([float(next_state[0]), float(next_state[1]), self.z0], self.q0),
                                            ReferenceType.Absolute),
                            asynchronous=True)

        # tmc = TwistStamped(); tmc.header = pm.header
        # tmc.twist.linear.x, tmc.twist.linear.y, tmc.twist.linear.z = vx, vy, 0.0
        # self.pub_cmd_twist.publish(tmc)

        now = rospy.Time.now()
        cs = self.robot.current_cartesian_state
        ee_pose = cs.pose.end_effector_pose
        ee_tw   = cs.velocity.end_effector_twist
        print ("ee_pose", ee_pose)

        # next_state[0] = float(ee_pose.translation[0])
        # next_state[1] = float(ee_pose.translation[1])
        # next_state[2] = float(ee_tw.linear[0])
        # next_state[3] = float(ee_tw.linear[1])
        print ("model state: ", next_state)
        print ("robot state: ", float(ee_pose.translation[0]), float(ee_pose.translation[1]), float(ee_tw.linear[0]), float(ee_tw.linear[1]))

        # float(ee_pose.translation[1]), float(ee_tw.linear[0]), float(ee_tw.linear[1])])

        # next_state = np.array([float(ee_pose.translation[0]), float(ee_pose.translation[1]), float(ee_tw.linear[0]), float(ee_tw.linear[1])])
        curr_state_msg = Float64MultiArray()
        curr_state_msg.data = curr_x.tolist()
        # Publish the state message
        self.curr_state_pub.publish(curr_state_msg)
        
        # Prepare the state message
        state_msg = Float64MultiArray()
        state_msg.data = next_state.tolist()
        # rospy.loginfo(f"SIM >> curr state: {curr_x}, next state: {state_msg.data}, control: {u}")
        # Publish the state message
        self.state_pub.publish(state_msg)
        
        cmd_arr = Float64MultiArray()
        cmd_arr.data = u.tolist()
        self.pub_cmd_arr.publish(cmd_arr)

        pm = PoseStamped()
        pm.header.stamp = now; pm.header.frame_id = self.frame_id
        pm.pose.position.x = float(ee_pose.translation[0])
        pm.pose.position.y = float(ee_pose.translation[1])
        pm.pose.position.z = float(ee_pose.translation[2])
        pm.pose.orientation.x = float(ee_pose.quaternion[0])
        pm.pose.orientation.y = float(ee_pose.quaternion[1])
        pm.pose.orientation.z = float(ee_pose.quaternion[2])
        pm.pose.orientation.w = float(ee_pose.quaternion[3])
        self.pub_pose.publish(pm)

        tm = TwistStamped(); tm.header = pm.header
        tm.twist.linear.x  = float(ee_tw.linear[0])
        tm.twist.linear.y  = float(ee_tw.linear[1])
        tm.twist.linear.z  = float(ee_tw.linear[2])
        tm.twist.angular.x = float(ee_tw.angular[0])
        tm.twist.angular.y = float(ee_tw.angular[1])
        tm.twist.angular.z = float(ee_tw.angular[2])
        self.pub_twist.publish(tm)


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


