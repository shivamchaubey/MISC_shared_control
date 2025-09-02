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
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import polytope

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root) +'/')
from config import config

from helper import _project_poly_H, _project_Fx_list, inflate_obstacle_poly, deflate_workspace_poly


def resize_polyhedron(cfg):
    ## DURING CIS computation we considered the margin of the robot size, and the orignal constrants are transfromed
    ## but for visualization we need to transform it to get the original shape.
    ## plot polytopes in X-Y coordinates
    ws_Fx = cfg.Fx.copy()
    obs_Fx = cfg.obs_Fx.copy()
    ws_fx = cfg.fx.copy()
    obs_fx = cfg.obs_fx.copy()

    ## need to fix the size as it's modified as per the agent size
    # but in real case it should be visible as original size
    act_obs = [deflate_workspace_poly(obs_Fx[i], obs_fx[i], cfg.agent_size) for i in range(len(obs_Fx))]
    obs_Fx = [act_obs[i][0] for i in range(len(act_obs))]
    obs_fx = [act_obs[i][1] for i in range(len(act_obs))]

    ## inflate obstacle (computation of cis is done via deflating it so we need to inflate for original space)
    act_ws = inflate_obstacle_poly(ws_Fx, ws_fx, cfg.agent_size)
    ws_Fx = act_ws[0]; ws_fx = act_ws[1]
    
    env_Fx = obs_Fx.copy(); env_fx = obs_fx.copy()
    env_Fx.append(ws_Fx); env_fx.append(ws_fx)
    return env_Fx, env_fx

class plotter():
    def __init__(self, cfg):


        self.robot_radius = cfg.agent_size
        
        # For demo purposes, I'll set dummy values for limits
        self.x_min = np.array([0, 0.5, -0.05, -0.05])
        self.x_max = np.array([1, 1, 0.05, 0.05])
        self.u_min = np.array([-1.0, -1.0])
        self.u_max = np.array([1.0, 1.0])

        self.per = 0.1
        
        env_Fx, env_fx = resize_polyhedron(cfg)
        self._Fp_list, self._fp_list = _project_Fx_list(env_Fx, env_fx, axes=(0, 1))



    def plot_pose(self, ax, xdata=[], ydata=[], per=0.1):        
        ax.clear()  # Clear axis for updating


        # ---- draw static polytopes ONCE here (kept static) ----
        self._poly_handles = []
        for i, (Fp, fp) in enumerate(zip(self._Fp_list, self._fp_list)):
            P = polytope.Polytope(Fp, fp)
            # Matplotlib collection returned; store handle so we can keep it
            coll = P.plot(ax=ax,
                          color='red' if i < len(self._Fp_list)-1 else 'green',
                          alpha=0.2,
                          edgecolor='red' if i < len(self._Fp_list)-1 else 'green')
            self._poly_handles.append(coll)

        ### Visualize robot
        traj, = ax.plot(xdata, ydata, '-b')
        pose, = ax.plot(xdata, ydata, '-o')

        # Create a button to show if the problem is feasible: a circular "button" outside the graph
        fig = ax.get_figure()
        button_ax = fig.add_axes([0.5, 0.92, 0.1, 0.1])  
        button_ax.set_xlim(-1, 1)
        button_ax.set_ylim(-1, 1)
        button_ax.axis('off')  
        button_circle = patches.Circle((0, 0), radius=0.2, edgecolor='black', facecolor='blue', linewidth=2)
        button_ax.add_patch(button_circle)

        ## robot visualization in a disk 
        self._disk_radius = float(self.robot_radius)
        robot = patches.Circle((0.2, 0.2),
                                    radius=self._disk_radius,
                                    facecolor='blue', edgecolor='blue',
                                    alpha=1, lw=1.5,
                                    animated=True, zorder=3)
        ax.add_patch(robot)


        # ax.set_xlim([self.x_min[0]*(1 - per), self.x_max[0]*(1 + per)])
        # ax.set_ylim([self.x_min[1]*(1 - per), self.x_max[1]*(1 + per)])

        # automatically set limits
        xs, ys = [], []
        if self._Fp_list:
            for Fp, fp in zip(self._Fp_list, self._fp_list):
                P = polytope.Polytope(Fp, fp)
                lo, hi = np.asarray(P.bounding_box[0]).ravel(), np.asarray(P.bounding_box[1]).ravel()
                xs += [lo[0], hi[0]]; ys += [lo[1], hi[1]]
            if xs:
                pad_x = 0.05 * (max(xs) - min(xs) + 1e-9)
                pad_y = 0.05 * (max(ys) - min(ys) + 1e-9)
                ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
                ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('State trajectory')

        return traj, pose, button_circle, robot

    def plot_velocity(self, ax, per=0.1):        
        ax.clear()  # Clear axis for updating
        xdata = []; ydata = []
        vel_x, = ax.plot(xdata, ydata, '-b', label='Velocity x')
        vel_y, = ax.plot(xdata, ydata, '-g', label='Velocity y')
        ax.set_xlabel('vx')
        ax.set_ylabel('vy')
        # ax.set_title('Velocity trajectory')
        ax.set_xlim([0, 100])
        ax.set_ylim([np.min(self.x_min[2:])*(1 + per), np.max(self.x_max[2:])*(1 + per)])
        # ax.legend()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
        fig = ax.get_figure()
        
        # To place the title above the legend:
        fig.text(0.83, 0.99, 'Velocity trajectory', ha='center', va='center', fontsize=12, fontweight='bold')

        ax.grid()
        return vel_x, vel_y

    def plot_control(self, ax, per=0.1):
        ax.clear()  # Clear axis for updating
        xdata = []; ydata = []
        ax_ass, = ax.plot(xdata, ydata, '-b', label='Assistance ax')
        ay_ass, = ax.plot(xdata, ydata, '-g', label='Assistance ay')  
        ax_user, = ax.plot(xdata, ydata, '--b', label='User ax', linewidth = 4, alpha = 0.5)
        ay_user, = ax.plot(xdata, ydata, '--g', label='User ay', linewidth = 4, alpha = 0.5)  
        ax.set_xlabel('ax')
        ax.set_ylabel('ay')
        # ax.set_title('Control Action', loc='center', pad=20)
        ax.set_xlim([0, 100])
        ax.set_ylim([np.min(self.u_min)*(1 + per), np.max(self.u_max)*(1 + per)])
        # ax.legend()
        # Move legend outside the plot, with 2 rows and 2 columns
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        # Create a title using fig.text (outside the axes, above the legend)
        fig = ax.get_figure()
        
        # To place the title above the legend:
        fig.text(0.83, 0.52, 'Control Action', ha='center', va='center', fontsize=12, fontweight='bold')

        ax.grid()
        return ax_ass, ay_ass, ax_user, ay_user

    def plot_combined(self, xdata=[], ydata=[], per=0.1, rotate_pose_180 = False):
        # Create figure and GridSpec layout
        fig = plt.figure(figsize=(10, 6))
        # plt.ion()
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])

        # Create axes for each plot
        ax_pose = fig.add_subplot(gs[:, 0])  # Left plot spanning both rows
        ax_velocity = fig.add_subplot(gs[0, 1])  # Top-right plot
        ax_control = fig.add_subplot(gs[1, 1])  # Bottom-right plot

        # Call individual plotter functions
        traj, pose, button_ax, robot = self.plot_pose(ax_pose, xdata, ydata, per)
        vel_x, vel_y = self.plot_velocity(ax_velocity, per)
        ax_ass, ay_ass, ax_user, ay_user = self.plot_control(ax_control, per)

        plt.tight_layout()
        # plt.show()

        if rotate_pose_180:
            # flip both axes -> visual 180° rotation
            xlo, xhi = ax_pose.get_xlim()
            ylo, yhi = ax_pose.get_ylim()
            ax_pose.set_xlim(xhi, xlo)
            ax_pose.set_ylim(yhi, ylo)
            # (optional) keep geometry nice
            ax_pose.set_aspect('equal', adjustable='box')


        return fig, plt, ax_pose, ax_velocity, ax_control, traj, pose, button_ax, robot, vel_x, vel_y, ax_ass, ay_ass, ax_user, ay_user
    


if __name__ == "__main__":
    import config
    cfg = config.config()
    plotter_ = plotter(cfg)
    fig, plt, ax_pose, ax_velocity, ax_control, traj, pose, button_ax, robot, vel_x, vel_y, ax_ass, ay_ass, ax_user, ay_user = plotter_.plot_combined()
    plt.show()