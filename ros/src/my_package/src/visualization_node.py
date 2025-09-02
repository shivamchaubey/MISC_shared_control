#!/usr/bin/env python

import os
# Remove paths that make Qt pick cv2’s plugins
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)

# Tell Qt to use PyQt5’s plugin dir
from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

# Optional: force xcb platform
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Now choose the Qt backend for Matplotlib
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.ion()

# Import cv2 *after* setting the plugin path so it can't override it
import cv2

import rospy
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from std_msgs.msg import Bool

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

import matplotlib.transforms as transforms
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import random
import time
import sys, os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
print (project_root)
sys.path.insert(0, str(project_root) +'/')
import config as cfg
cfg = cfg.config()
sys.path.insert(1, cfg.lib_path)
from visualization import Plotter

class ROSPlotter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('PlotterNode')

        # Set up the rate for publishing (e.g., 10 Hz)
        self.rate = rospy.Rate(150)  # 10 Hz (0.1 seconds)

        # Initialize subscriber to /state topic
        self.state_sub = rospy.Subscriber('/state', Float64MultiArray, self.state_callback)

        # Initialize subscriber to /assistive_control topic
        self.control_assistive_sub = rospy.Subscriber('/assistive_input', Float64MultiArray, self.control_assistive_callback)

        # # Initialize subscriber to /user_control topic
        self.control_user_sub = rospy.Subscriber('/user_input', Float32MultiArray, self.control_user_callback)

        # # Initialize subscriber to /problem_feasible topic
        self.control_feasibilty = rospy.Subscriber('/problem_feasible', Bool, self.problem_feasible_callback)

        # Variables to hold the latest state and control data
        self.state = None
        self.assistive_control = None
        self.user_control = None
        self.feasible = None

        # for plotter
        rotate_pose_180 = True
        self.visual = Plotter(cfg)
        plt.ion()
        self.fig, self.plot, self.ax_pose, self.ax_vel, self.ax_ctrl, self.traj, self.pose, self.feas_button, self.robot, self.vel_x, self.vel_y, self.ctrl_x_ass, self.ctrl_y_ass, self.ctrl_x_user, self.ctrl_y_user = self.visual.plot_combined(rotate_pose_180 = rotate_pose_180)

        plt.show()
        self.x_hist = []
        self.y_hist = []
        self.vx_hist = []
        self.vy_hist = []
        self.ax_ass_hist = []
        self.ay_ass_hist = []
        self.ax_user_hist = []
        self.ay_user_hist = []
        
        self.hist_thres = 100
        self.per = 0.1
        self.state_counter = 1
        self.ctrl_counter = 1

    def state_callback(self, msg):
        """Callback function to handle state updates."""
        self.state = msg.data
        # rospy.loginfo(f"Received state: {self.state}")

    def problem_feasible_callback(self, msg):
        """Callback function to handle state updates."""
        self.feasible = msg.data
        # rospy.loginfo(f"Problem Feasible: {self.feasible}")

    def control_assistive_callback(self, msg):
        """Callback function to handle control input updates."""
        self.assistive_control = msg.data
        # rospy.loginfo(f"Received assistive control: {self.assistive_control}")

    def control_user_callback(self, msg):
        """Callback function to handle control input updates."""
        self.user_control = msg.data
        # rospy.loginfo(f"Received user control: {self.user_control}")
    
    def plot_update(self, update_axilim = False):
        # print ("plot_update")
        plt.ion()
        if self.state !=None: 

            ## plot pose
            [x, y, vx, vy] = self.state
            self.x_hist.append(x)
            self.y_hist.append(y)
            self.robot.center = (x, y)
            self.traj.set_data(self.x_hist, self.y_hist)
            self.pose.set_data(self.x_hist[-1], self.y_hist[-1])

            # self.pose.set_ydata(self.y_hist)
            x_limits = self.ax_pose.get_xlim()
            y_limits = self.ax_pose.get_ylim()
            x_lim = [np.min(self.x_hist), np.max(self.x_hist)]
            y_lim = [np.min(self.y_hist), np.max(self.y_hist)]
            
            x_lim_min = min([x_limits[0], x_lim[0]])
            x_lim_max = max([x_limits[1], x_lim[1]])

            y_lim_min = min([y_limits[0], y_lim[0]])
            y_lim_max = max([y_limits[1], y_lim[1]])

            if self.feasible != None:
                if self.feasible:
                    self.feas_button.set_facecolor('green')
                    # plt.draw()

                else:
                    self.feas_button.set_facecolor('red')
                    # plt.draw()
                plt.draw()
                # self.feas_button.get_figure().canvas.draw()
                # self.feas_button.get_figure().canvas.flush_events()

            if update_axilim:
                self.ax_pose.set_xlim(x_lim_min, x_lim_max)
                self.ax_pose.set_ylim(y_lim_min, y_lim_max)

            # Redraw the pose plot canvas
            self.ax_pose.get_figure().canvas.draw_idle()
            self.ax_pose.get_figure().canvas.flush_events()

            ## plot velocity
            self.vx_hist.append(vx)
            self.vy_hist.append(vy)
            # print ("self.vx_hist", self.vx_hist)
            self.vel_x.set_data(np.arange(self.state_counter,self.state_counter + len(self.vx_hist)), self.vx_hist)
            self.vel_y.set_data(np.arange(self.state_counter,self.state_counter + len(self.vy_hist)), self.vy_hist)
            # print ("self.state_counter, self.state_counter + len(self.vx_hist)", self.state_counter, self.state_counter + len(self.vx_hist))
            self.ax_vel.set_xlim([self.state_counter, self.state_counter + len(self.vx_hist)])
            if update_axilim:
                y_lim_min, y_lim_max = min([np.min(self.vx_hist), np.min(self.vy_hist)]), max([np.max(self.vx_hist), np.max(self.vy_hist)])
                self.ax_vel.set_ylim(y_lim_min*(1-self.per), y_lim_max*(1+self.per))
            

            self.state_counter += 1

        ## plot control
        if self.assistive_control != None and self.user_control !=None: 
            [ax_ass, ay_ass] = self.assistive_control
            [ax_user, ay_user] = self.user_control

            self.ax_ass_hist.append(ax_ass)
            self.ay_ass_hist.append(ay_ass)
            self.ctrl_x_ass.set_data(np.arange(self.ctrl_counter,self.ctrl_counter + len(self.ax_ass_hist)), self.ax_ass_hist)
            self.ctrl_y_ass.set_data(np.arange(self.ctrl_counter,self.ctrl_counter + len(self.ay_ass_hist)), self.ay_ass_hist)
            self.ax_user_hist.append(ax_user)
            self.ay_user_hist.append(ay_user)
            self.ctrl_x_user.set_data(np.arange(self.ctrl_counter,self.ctrl_counter + len(self.ax_user_hist)), self.ax_user_hist)
            self.ctrl_y_user.set_data(np.arange(self.ctrl_counter,self.ctrl_counter + len(self.ay_user_hist)), self.ay_user_hist)

            self.ax_ctrl.set_xlim([self.ctrl_counter, self.ctrl_counter + len(self.ax_ass_hist)])
            if update_axilim:
                ay_lim_min = min([np.min(self.ax_ass_hist), np.min(self.ay_ass_hist), np.min(self.ax_user_hist), np.min(self.ay_user_hist)])
                ay_lim_max = max([np.max(self.ax_ass_hist), np.max(self.ay_ass_hist), np.max(self.ax_user_hist), np.max(self.ay_user_hist)])
                self.ax_ctrl.set_ylim(ay_lim_min*(1-self.per), ay_lim_max*(1+self.per))
            
            ## update together to have same grid line
            # Redraw the velocity plot canvas
            self.ax_vel.get_figure().canvas.draw_idle()
            self.ax_vel.get_figure().canvas.flush_events()
            # Redraw the control plot canvas
            self.ax_ctrl.get_figure().canvas.draw_idle()
            self.ax_ctrl.get_figure().canvas.flush_events()


            # Redraw the main figure canvas
            # self.fig.canvas.draw_idle()
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            # plt.show()
            time.sleep(0.001)
            if len(self.x_hist) >= self.hist_thres:
                self.x_hist.pop(0)
                self.y_hist.pop(0)
                self.vx_hist.pop(0)
                self.vy_hist.pop(0)
                self.ax_ass_hist.pop(0)
                self.ay_ass_hist.pop(0)
                self.ax_user_hist.pop(0)
                self.ay_user_hist.pop(0)
    
            # self.plot.show(block = False)

            self.ctrl_counter += 1

    def spin(self):
        """Keep the node running and processing callbacks."""
        while not rospy.is_shutdown():
            # Continuously publish the state at a set frequency
            self.plot_update()
            self.rate.sleep()  # Sleep to maintain the loop at the desired frequency


if __name__ == '__main__':
    try:
        # Create an instance of the subscriber node and start it
        node = ROSPlotter()
        node.spin()
    except rospy.ROSInterruptException:
        pass
