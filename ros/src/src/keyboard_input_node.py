#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

import curses
import math

import rospy
from std_msgs.msg import Float32MultiArray


class TextWindow():

    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError('lineno out of bounds')
        height, width = self._screen.getmaxyx()
        y = int((height / self._num_lines) * lineno)
        x = 10
        for text in message.split('\n'):
            text = text.ljust(width)
            # print ("text",text)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()


class KeyTeleop():

    _interface = None

    _linear = None
    _angular = None

    def __init__(self, interface):
        self._interface = interface

        # Initialize the ROS node
        rospy.init_node('KeyboardInputNode')

        self.verbose = False
        # Set up the rate for publishing (e.g., 10 Hz)
        self.rate = rospy.Rate(180)  # 10 Hz (0.1 seconds)

        self._num_steps = 1.0
        self._num_steps2 = 1.0

        ##################### control command publisher ######################
        # user control input
        self.control_pub = rospy.Publisher('/user_input', Float32MultiArray, queue_size=1)

    def run(self):
        self._linear = 0
        self._angular = 0

        # Publish message for the corrected control input [ax, ay]
        control_msg = Float32MultiArray()
        control_msg.data = [-self._angular, self._linear]
                    
        while True:
            keycode = self._interface.read_key()
            control_msg = Float32MultiArray()
            control_msg.data = [-self._angular, self._linear]

            self._interface.clear()
            self._interface.write_line(2, 'LEFT/RIGHT: %f, UP/DOWN : %f' % (-self._angular, self._linear))
            self._interface.write_line(5, 'Use arrow keys to move, space to stop, q to exit.')
            self._interface.refresh()
            if self.verbose :
                rospy.loginfo(f"Received user control: {control_msg.data}")

            if keycode:
                if self._key_pressed(keycode):    
                    self.control_pub.publish(control_msg)
                else:
                    control_msg.data = [0, 0]
                    self.control_pub.publish(control_msg)
            else:
                # print ("keycode is None")

                # control_msg = Float32MultiArray()
                # control_msg.data = [0, 0]
                self.control_pub.publish(control_msg)
            if keycode == ord('q'):
                break

            self.rate.sleep()



    def _get_twist(self, linear, angular):
        twist = Twist()
        if linear >= 0:
            twist.linear.x = self._forward(1.0, linear)
        else:
            twist.linear.x = self._backward(-1.0, -linear)
        twist.angular.z = self._rotation(math.copysign(1, angular), abs(angular))
        return twist

    def _key_pressed(self, keycode):
        dt = 0.01
        movement_bindings = {
            curses.KEY_UP:    ( 1.0,  0.0),
            curses.KEY_DOWN:  (-1.0,  0.0),
            curses.KEY_LEFT:  ( 0.0, 0.1),
            curses.KEY_RIGHT: ( 0.0, -0.1),
        }
        speed_bindings = {
            ord(' '): (0.0, 0.0),
        }
        if keycode in movement_bindings:
            acc = movement_bindings[keycode]
            
            ok = False
            if acc[0]:
                linear = self._linear + acc[0]*dt*1.0
                if abs(linear) <= self._num_steps:
                    self._linear = linear
                    ok = True
            if acc[1]:
                angular = self._angular + acc[1]*dt*5.0
                if abs(angular) <= self._num_steps2:
                    self._angular = angular
                    ok = True
            if not ok:
                self._interface.beep()

        elif keycode in speed_bindings:
            acc = speed_bindings[keycode]
            # Note: bounds aren't enforced here!
            if acc[0] is not None:
                self._linear = acc[0]
            if acc[1] is not None:
                self._angular = acc[1]
        # else:
        #     acc = [0.0, 0.0]
        #     print ("space bar ...........")
        #     # acc = acc
        #     # Note: bounds aren't enforced here!
        #     if acc[0] is not None:
        #         self._linear = acc[0]
        #     if acc[1] is not None:
        #         self._angular = acc[1]

        if keycode == ord('q'):
            self._running = False
            rospy.signal_shutdown('Bye')
        else:
            return False

        return True

    def _publish(self):
        self._interface.clear()
        self._interface.write_line(2, 'Acceleration: %f, Steering: %f' % (self._linear, self._angular))
        self._interface.write_line(5, 'Use arrow keys to move, space to stop, q to exit.')
        self._interface.refresh()


        twist = self._get_twist(self._linear, self._angular)
        # self._pub_cmd.publish(twist)
        
        self.time1 = rospy.get_time()
        # print ("time at start publishing",self.time1-self.time0)

        if  (self.past_linear != self._linear): 
        
            self.accel_commands.publish(self._linear)

        if (self.past_angular != self._angular): 
 
            self.steering_commands.publish(self._angular)

        self.past_linear = self._linear 
        self.past_angular = self._angular 
        
        # self.imu_vx.publish(self.imu_enc.vx)
        # self.imu_vy.publish(self.imu_enc.vy)
        # self.imu_X.publish(self.imu_enc.X)
        # self.imu_Y.publish(self.imu_enc.Y)
        # self.imu_ax.publish(self.imu_enc.ax)
        # self.imu_ay.publish(self.imu_enc.ay)
        # self.imu_az.publish(self.imu_enc.az)

        # self.roll.publish(self.imu_enc.roll*180.0/pi)
        # self.yaw.publish(self.imu_enc.yaw*180/pi)
        # self.pitch.publish(self.imu_enc.pitch*180/pi)

        # # print ("time for finishing publishing",rospy.get_time()-self.time1)

        # self.control_commands_his["real_timestamp_ms"]. append(rospy.get_time())
        # self.control_commands_his["timestamp_ms"]. append(rospy.get_time()-self.time0)
        # self.control_commands_his["acceleration"]. append(self._linear)
        # self.control_commands_his["steering"]. append(self._angular)

def main(stdscr):
    # rospy.init_node('keyboard_control')


    # app = SimpleKeyTeleop(TextWindow(stdscr))
    app = KeyTeleop(TextWindow(stdscr))
    app.run()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass
