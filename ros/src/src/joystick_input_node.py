#!/usr/bin/env python3

#ManhLinhPhan


import rospy
import math
from sensor_msgs.msg import JoyFeedbackArray, Joy
from std_msgs.msg import Float32MultiArray
import numpy as np
from pydualsense import *
import sys, os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root) +'/')
import config as cfg
cfg = cfg.config()
sys.path.insert(1, cfg.python_lib_path)


class Ds5Ros():
    def __init__(self):

        self.logging_prefix = "DS5_ROS: "
        
        # 0 is startup, 1 is running
        self.node_state = 0

        # create dualsense
        self.dualsense = pydualsense()

        self.haptic_feedback = False
        self.rate = cfg.joystick_frequency 
        # receive feedback from robot_joy_control node 
        self.joy_sub_topic = rospy.get_param("joy_sub", "joy/set_feedback")
        # send signal to robot_joy_control node
        self.joy_pub_topic = rospy.get_param("joy_pub", "joy")

        # controller is not straight zero for the axis.. prevent tiny robot movements 
        self.deadzone = rospy.get_param("deadzone", 0.05)
        self.scale = cfg.scale_factor

        self.joy_sub = rospy.Subscriber(self.joy_sub_topic, JoyFeedbackArray, self.set_feedback, queue_size= 1)
        self.joy_pub = rospy.Publisher(self.joy_pub_topic, Joy, queue_size = 1)

        # user control input
        self.control_pub = rospy.Publisher('/user_input', Float32MultiArray, queue_size=1)
        # Initialize subscriber for assistive system input input
        # self.control_sub = rospy.Subscriber('/assistive_input', Float32MultiArray, self.control_callback)
        self.u = None
        self.user_input = None
        
        self.maskR = 0xFF0000
        self.maskG = 0x00FF00
        self.maskB = 0x0000FF
    '''
    JoyFeedback Struct
    TYPE_LED = 0 | TYPE_RUMBLE = 1 (use also for feedback on rear button) | TYPE_BUZZER = 2
    type    : above
    id      : 0 = Left Motor Rumble | 1 = Right Motor Rumble | 2 = L2 feedback Trigger | 3 = R2 feedback Trigger
    Using combination of type and id to get the result
    intensity:  
    '''
    def control_callback(self, msg):
        """Callback function to handle control input updates."""
        # Extract control input from the received message
        self.u = msg.data # Update control input with received data
        if self.u != None and self.user_input != None and self.haptic_feedback:
            val = 0
            u_ass = np.array(self.u) 
            u_user = np.array(self.user_input)
            thresh = 0.40
            if u_user[0] != 0.0:
                per_x = ((abs(u_user[0]) - abs(u_ass[0]))/abs(u_user[0]))/2
                print ("per_x", per_x)
                if per_x >= thresh:
                    per_x = np.clip(per_x, 0, 1)
                    val = ((per_x - thresh)/thresh)*255
                    self.dualsense.setLeftMotor(int(val))
                    # self.dualsense.triggerL.setMode(TriggerModes.Rigid)
                    # self.dualsense.triggerL.setForce(1, 240)
            if u_user[1] != 0.0:
                per_y = ((abs(u_user[1]) - abs(u_ass[1]))/abs(u_user[1]))/2
        # rospy.loginfo(f"Received control input: {self.u}")

                print ("per_y", per_y)
                if per_y >= thresh:
                    per_y = np.clip(per_y, 0, 1)
                    # val = per_y*255
                    val = ((per_y - thresh)/thresh)*255
                    self.dualsense.setRightMotor(int(val))
            self.dualsense.setLeftMotor(int(val))
            self.dualsense.setRightMotor(int(val))

                    # self.dualsense.triggerR.setMode(TriggerModes.Rigid)
                    # self.dualsense.triggerR.setForce(1, 240)


    def set_feedback(self, msg):
        #print(msg)
        for feedback in msg.array: #not iterable
            # type = 0: LED control
            # RGB (24bit) is sent as float32 -> can almost be covered by 23 mantissa  
            if feedback.type == 0:
                
                int_intensity = int(feedback.intensity)
                light_red = (int_intensity & self.maskR) >> 16
                light_green = (int_intensity & self.maskG) >> 8
                light_blue = (int_intensity & self.maskB)

                self.dualsense.light.setColorI(light_red,light_green,light_blue)
                continue
            
            #other type
            if feedback.intensity > 1.0 or feedback.intensity <0.0:
                rospy.logerr(self.logging_prefix + 'intensity must be in range 0.0 - 1.0')
                continue
            else:
                feedback.intensity = feedback.intensity * 255.0
            
            #intensity muss be an integer
            if feedback.type == 1 and feedback.id == 0:
                pass #add effect later
            elif feedback.type == 1 and feedback.id == 1:
                pass #add effect later
            elif feedback.type == 1 and feedback.id == 2:
                self.dualsense.triggerL.setMode(TriggerModes.Rigid)
                self.dualsense.triggerL.setForce(1, int(feedback.intensity))
            elif feedback.type == 1 and feedback.id == 3:
                self.dualsense.triggerR.setMode(TriggerModes.Rigid)
                self.dualsense.triggerR.setForce(1, int(feedback.intensity) )

    def joy_publish(self):
        # try:
        print ("publishing joy message")
        joy_msg = Joy()

        #print("in joy_publish")
        
        for i in range(18):
            joy_msg.buttons.append(0)
        # Buttons mapping
        # print ("self.dualsense.state.cross", self.dualsense.state.cross)
        joy_msg.buttons[0] = self.dualsense.state.cross
        joy_msg.buttons[1] = self.dualsense.state.circle
        joy_msg.buttons[2] = self.dualsense.state.triangle
        joy_msg.buttons[3] = self.dualsense.state.square

        joy_msg.buttons[13] = self.dualsense.state.DpadUp
        joy_msg.buttons[16] = self.dualsense.state.DpadRight
        joy_msg.buttons[14] = self.dualsense.state.DpadDown
        joy_msg.buttons[15] = self.dualsense.state.DpadLeft

        joy_msg.buttons[6] = self.dualsense.state.L2Btn
        joy_msg.buttons[7] = self.dualsense.state.R2Btn
        joy_msg.buttons[4] = self.dualsense.state.L1
        joy_msg.buttons[5] = self.dualsense.state.R1
        joy_msg.buttons[8] = self.dualsense.state.L3
        joy_msg.buttons[9] = self.dualsense.state.R3
        
        joy_msg.buttons[10] = self.dualsense.state.ps
        joy_msg.buttons[11] = self.dualsense.state.share
        joy_msg.buttons[12] = self.dualsense.state.options

        joy_msg.buttons[17] = self.dualsense.state.touchBtn
        # print ("joy_msg.buttons", joy_msg.buttons)
        # Axes mapping
        for i in range(6):
            joy_msg.axes.append(0)

        # change value by Axes from 0.0 -> 1.0
        joy_msg.axes[0] = (-1 * self.dualsense.state.LX )/128.0         #Leftward   (-1.0 -> 1.0, default ~0.0)
        joy_msg.axes[1] = (-1 * self.dualsense.state.LY )/128.0         #Upward     (-1.0 -> 1.0, default ~0.0)
        joy_msg.axes[2] = (-1 * self.dualsense.state.RX )/128.0         #Leftward   (-1.0 -> 1.0, default ~0.0)
        joy_msg.axes[3] = (-1 * self.dualsense.state.RY )/128.0         #Upward     (-1.0 -> 1.0, default ~0.0)
        joy_msg.axes[4] = self.dualsense.state.L2 /255.0                #PushDown   (0.0 -> 1.0, default = 0)
        joy_msg.axes[5] = self.dualsense.state.R2 /255.0                #PushDown   (0.0 -> 1.0, default = 0)
        # print ("joy_msg.axes", joy_msg.axes)

        for val in range(6):
            if(abs(joy_msg.axes[val]) < self.deadzone):
                joy_msg.axes[val] = 0.0
        
        # Create a message for the user control input [ax, ay]
        control_msg = Float32MultiArray()
        # control_msg.data = [-joy_msg.axes[0], joy_msg.axes[3]]
        control_msg.data = [-self.scale*joy_msg.axes[2], self.scale*joy_msg.axes[3]]
        self.user_input = control_msg.data
        self.control_pub.publish(control_msg)

        self.joy_pub.publish(joy_msg)
        # except:
        #     print("catch error")

    # def feedback_publish(self):
        # from pydualsense import *

# # get dualsense instance
# dualsense = pydualsense()
# dualsense.init()

# print('Trigger Effect demo started')

# dualsense.setLeftMotor(255)
# # dualsense.setRightMotor(100)
# # dualsense.triggerL.setMode(TriggerModes.Rigid)
# dualsense.triggerL.setMode(TriggerModes.Rigid_A)
# # dualsense.triggerL.setMode(TriggerModes.Pulse_A)

# dualsense.triggerL.setForce(1, 255)

# # dualsense.triggerR.setMode(TriggerModes.Pulse_A)
# # dualsense.triggerR.setForce(0, 200)
# # dualsense.triggerR.setForce(1, 255)
# # dualsense.triggerR.setForce(2, 175)

# # loop until r1 is pressed to feel effect
# while not dualsense.state.R1:
#     ...
# # terminate the thread for message and close the device
# dualsense.close()

    def main_loop(self):
        rate = rospy.Rate(self.rate)

        ### initialize controller if possible, else wait
        print ("### initialize controller if possible, else wait")
        
        # find device and initialize
        while not rospy.is_shutdown():
            print ("while loop", "self.node_state", self.node_state)
            if self.node_state == 0:
                try:
                    self.dualsense.init()
                    print("DS5_Ros is initialized!")
                except:
                    rospy.logwarn_throttle(2, "Cannot initialize controller!")
                else:
                    # set rgb led to green
                    self.dualsense.light.setColorI(0,255,0)
                    self.node_state = 1
            
            elif self.node_state == 1:
                # if error -> node_state = 0
                try:
                    #Add new attribute cable_connection in dualsense
                    #Init cable_connection = True
                    #Catch IOError in dualsense.writeReport and set cable_connection = False
                    print ("self.dualsense.conType", self.dualsense.conType!=None)
                    if self.dualsense.conType!=None:
                        # print ("self.dualsense.cable_connection", self.dualsense.cable_connection)
                        rospy.loginfo_throttle(2, "DS5_Ros is alive!" )
                        self.joy_publish()
                    else:
                        print ("else")
                        rospy.logerr("Lost connection! Go back to init!")
                        self.node_state = 0 
                except:
                    print ("pass")
                    pass

            rate.sleep()
        print ("self.dualsense.close()")
        try:
            self.dualsense.close()
        except:
            pass

if __name__== '__main__':
    rospy.init_node("ds5ros_node")
    ds5 = Ds5Ros()
    ds5.main_loop()