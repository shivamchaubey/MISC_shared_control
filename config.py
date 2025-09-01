# config.py
import sys, os
import numpy as np

class config:
    def __init__(self):
        
        ## experiment mode debug or user_study
        self.experiment_mode = 'user_study' # 'user_study' or 'debugging'
        # self.experiment_mode = 'debug' # 'user_study' or 'debugging'
        # if self.experiment_mode == 'debug':
        #     # self.number_of_goals = [0, 4] #5 #only for debugging
        #     self.number_of_constraints = [] #5 #only for debugging
        
        ## path settings
        import sys, os
        from pathlib import Path

        project_root = str(Path(__file__).resolve().parent)
        print ("project_root", project_root)
        # sys.path.insert(0, str(project_root) +'/')
        
        self.root_path = project_root + '/' 
        self.lib_path = self.root_path + 'lib'
        self.ros_path = self.root_path + 'ros'
        self.ros_setup_path = self.ros_path + "/devel/setup.bash"
        self.matlab_cis_path = "/home/shivam/Dropbox_Aalto/research_work/codes/assistive_system/dev/github_code/cis2m/matlab/"

        # user study:

        map_name = 'real_world_exp_dist.png'
        self.user_study_path = self.root_path + 'user_study'
        self.user_study_data_path = self.root_path + 'user_study/data'
        self.image_map_path = self.root_path + 'lib/extra/map/' + map_name

        ## game settings
        self.game_play_time = 60
        if self.experiment_mode == 'debug':
            self.game_train_time = 10
        else:
            self.game_train_time = 120
        self.game_play_mode = 'goal' # 'goal' or 'time'
        self.game_train_mode = 'time' # 'goal' or 'time'
        self.respawn_mode = 0 #1: for last state before entering unsafe region, 0 for last point it started 
        self.goal_size = 0.020 # diameter
        self.agent_size = 0.019 # diameter
        self.goal_vel_threshold = 0.9 # velocity threshold to account if goal reached
        # if self.experiment_mode == 'debug':
        #     self.record_data = False
        # else:
        #     self.record_data = True
        # self.record_data = True
        
        ## simulation, assitive system and joystick settings
        self.joystick_frequency = 20 # joystick publishing frequency
        self.simulation_frequency = 10 # simulation frequency [Note]: you will need to recalculate the Control invariant set if you change this value
        self.controller_frequency = 50 # controller frequency
        self.sys_dt = 1/self.simulation_frequency # system dynamics time step [Note]: you will need to recalculate the Control invariant set if you change this value
        self.damping_factor = 0.1  ## system dynamics damping factor # [Note]: you will need to recalculate the Control invariant set if you change this value
        self.N = 1 # MPC prediction horizon
        self.N += 1 # N: Horizon, +1 because of state x0
        self.max_vel = 0.05 # [Note]: you will need to recalculate the Control invariant set if you change this value
        self.max_accel = 1.0 #[Note]: you will need to recalculate the Control invariant set if you change this value
        #joystick
        self.deadzone = 0.05 #joystick deadzone
        self.scale_factor = 0.95 #joystick scale factor

        ## Control invariant set computation
        self.matlab_cis_function_path = "/home/shivam/Dropbox_Aalto/research_work/codes/assistive_system/dev/github_code/cis2m/matlab/"
        self.python_lib_path = self.lib_path
        self.compute_safe_set_cons = False

        ############ only used in simulation or while computing CIS ############
        self.disturbance = True
        self.E = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])

        self.Gw = np.array([[ 1.,  0.,  0.,  0.],
                             [ 0.,  1.,  0.,  0.],
                             [ 0.,  0.,  1.,  0.],
                             [ 0.,  0.,  0.,  1.],
                             [-1., -0., -0., -0.],
                             [-0., -1., -0., -0.],
                             [-0., -0., -1., -0.],
                             [-0., -0., -0., -1.]])
        
        self.Fw = np.array([[0.00220696731294516, 0.00033361141907932, 0.01511132840251176, 0.00392036854057386, 0.00137375336326578, 0.00060107734896421, 0.01921057797980761, 0.00721068247405685]]).T
        ##########################################################################

        ## visualization setting
        # self.loading_img_src="https://i.gifer.com/bf6.gif"
        self.loading_img_src="../static/images/loading.gif"
