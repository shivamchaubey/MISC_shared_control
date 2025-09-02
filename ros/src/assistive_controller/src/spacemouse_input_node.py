#!/usr/bin/env python
# ROS1 SpaceMouse publisher with a read() pump thread.

import sys, threading, time
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
import pyspacemouse

class SpaceMouseROS(object):
    def __init__(self):
        self.rate_hz       = rospy.get_param("~rate", 100)
        self.deadband      = rospy.get_param("~deadband", 0.02)
        self.scale_linear  = rospy.get_param("~scale_linear", 1.0)
        self.scale_angular = rospy.get_param("~scale_angular", 1.0)
        self.invert_axes   = rospy.get_param("~invert_axes", [1,1,1,1,1,1])

        self.axes = [0.0]*6
        self.buttons = []
        self.lock = threading.Lock()

        self.joy_pub   = rospy.Publisher("spacemouse/joy", Joy, queue_size=1)
        self.twist_pub = rospy.Publisher("spacemouse/twist", Twist, queue_size=1)
        self.control_pub = rospy.Publisher('/user_input', Float32MultiArray, queue_size=1)

        # Explicitly keep nonblocking mode on (needed for callbacks).
        ok = pyspacemouse.open(
            dof_callback=self._on_dof,
            button_callback=self._on_button,
            set_nonblocking_loop=True
        )
        if not ok:
            rospy.logfatal("Failed to open SpaceMouse.")
            raise RuntimeError("pyspacemouse.open() failed")
        rospy.loginfo("SpaceMouse opened.")

        # ---- pump thread calls read() so callbacks actually fire ----
        self._pump_alive = True
        self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread.start()

    def _pump_loop(self):
        # Poll as fast as practical; sleep a bit to be nice to CPU.
        while self._pump_alive and not rospy.is_shutdown():
            pyspacemouse.read()   # this triggers callbacks internally
            time.sleep(0.001)

    def _db(self, v):
        return 0.0 if abs(v) < self.deadband else v

    def _on_dof(self, state):
        # state has attributes x,y,z, roll,pitch,yaw in ~[-1,1]
        raw = [float(getattr(state, k)) for k in ["x","y","z","roll","pitch","yaw"]]
        # Optional normalize not needed (already ~[-1,1] in this fork), apply deadband and sign flips.
        proc = [self._db(r)*float(self.invert_axes[i]) for i, r in enumerate(raw)]
        with self.lock:
            self.axes = proc

    def _on_button(self, state, buttons):
        # buttons is a list of 0/1
        with self.lock:
            self.buttons = list(buttons)

    def spin(self):
        r = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                axes = list(self.axes)
                buttons = list(self.buttons) if self.buttons else [0,0]

            joy = Joy()
            joy.header.stamp = rospy.Time.now()
            joy.axes = axes
            joy.buttons = buttons
            self.joy_pub.publish(joy)

            control_msg = Float32MultiArray()
            control_msg.data = [-joy.axes[0], -joy.axes[1]]
            rospy.loginfo("Control message: %s", control_msg.data)
            self.control_pub.publish(control_msg)

            tw = Twist()
            tw.linear.x, tw.linear.y, tw.linear.z   = [a*self.scale_linear for a in axes[0:3]]
            tw.angular.x, tw.angular.y, tw.angular.z = [a*self.scale_angular for a in axes[3:6]]
            self.twist_pub.publish(tw)
            r.sleep()

    def shutdown(self):
        self._pump_alive = False
        try:
            pyspacemouse.close()
        except Exception:
            pass

def main():
    rospy.init_node("spacemouse_publisher")
    node = SpaceMouseROS()
    try:
        node.spin()
    finally:
        node.shutdown()

if __name__ == "__main__":
    main()
