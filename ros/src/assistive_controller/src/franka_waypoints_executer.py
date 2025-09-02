import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from franky import (
    Robot, Affine, ReferenceType,
    CartesianWaypoint, CartesianWaypointMotion,
    RobotPose, ElbowState, RelativeDynamicsFactor,
    CartesianState, Twist, Duration,
    Robot, JointWaypoint, JointWaypointMotion
)



# from franky import (
#     Robot, JointWaypoint, JointWaypointMotion
# )

# --- your waypoints ---
homing_position = np.array([0.695, 0.160, 0.40])
wp_pt1 = np.array([0.695, 0.160, 0.27])
wp_pt2 = np.array([0.695, 0.096, 0.27])
wp_pt3 = np.array([0.663, 0.096, 0.27])
wp_pt4 = np.array([0.663, 0.000, 0.27])
wp_pt5 = np.array([0.599, 0.000, 0.27])
wp_pt6 = np.array([0.599, 0.160, 0.27])
wp_pt7 = np.array([0.631, 0.160, 0.27])

WAYPOINTS = [
    homing_position,
    wp_pt1, wp_pt2, wp_pt3, wp_pt4, wp_pt5, wp_pt6, wp_pt7,
    homing_position
]

def make_quat_down():
    # Tool Z down; adjust if your tool frame differs
    return R.from_euler("xyz", [math.pi, 0.0, 0.0]).as_quat()  # x,y,z,w

def ensure_quat_continuity(quats):
    """Avoid 180° flips by flipping sign if dot<0 (q and -q are same rotation)."""
    qlist = [np.array(quats[0], dtype=float)]
    for q in quats[1:]:
        q = np.array(q, dtype=float)
        if np.dot(qlist[-1], q) < 0.0:
            q = -q
        qlist.append(q)
    return [q.tolist() for q in qlist]

def build_cartesian_waypoints(points_xyz, quat_list, elbow_state):
    wps = []
    for i, (p, q) in enumerate(zip(points_xyz, quat_list)):
        pose = RobotPose(Affine(np.asarray(p, float).tolist(), q),
                         elbow_state=elbow_state)
        # gentle dynamics on inner points to reduce sharp nullspace moves
        dyn = RelativeDynamicsFactor(0.6, 0.6, 0.6) if 0 < i < len(points_xyz)-1 else None
        if dyn:
            wps.append(CartesianWaypoint(pose, ReferenceType.Absolute, dyn))
        else:
            wps.append(CartesianWaypoint(pose))
    return wps

def main():
    robot = Robot("172.16.0.2")  # <-- set your IP
    robot.recover_from_errors()

    # Start slow overall, and additionally clamp elbow motion specifically.
    robot.relative_dynamics_factor = 0.04
    robot.elbow_velocity_limit.set(0.6)      # rad/s (tune up once stable)
    robot.elbow_acceleration_limit.set(5.0)  # rad/s^2
    robot.elbow_jerk_limit.set(1800.0)       # rad/s^3
    
    # define initial joint state (replace with your measured values!)
    initial_joint_pos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    # build a joint waypoint motion
    # joint_wp = JointWaypoint(initial_joint_pos)
    # joint_motion = JointWaypointMotion([joint_wp])

    # # execute
    # robot.move(joint_motion)

    N = 4
    for i in range(N):
        # Read the *current* elbow and lock it for the whole path
        cur_elbow = robot.current_cartesian_state.pose.elbow_state  # franky object
        # If you want a slight bias, you can do: ElbowState(cur_elbow.q + 0.0)

        # Build a continuous quaternion list (prevents orientation flips)
        q0 = make_quat_down()
        quat_list = ensure_quat_continuity([q0]*len(WAYPOINTS))

        waypoints = build_cartesian_waypoints(WAYPOINTS, quat_list, elbow_state=cur_elbow)
        motion = CartesianWaypointMotion(waypoints)
        
        # if i==0:

        #     # Save them
        #     save_franky_waypoints_npz("plan_A.npz", waypoints)

        #     # Later (or in another script), load them
        #     waypoints_loaded, motion = load_franky_waypoints_npz("plan_A.npz")

        #     save_motion_npz("traj_A.npz", motion)

        #     # Later
        #     motion_loaded = load_motion_npz("traj_A.npz")


        robot.move(motion)
    
    print("Done.")

if __name__ == "__main__":
    main()
