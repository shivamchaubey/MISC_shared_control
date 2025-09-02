# config.py
import sys, os
import numpy as np
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root) +'/lib')
from constraint_builder import ConstraintHelper
from CIS_computation import ExtractCIS
from helper import load_cis_npz, assert_has_any_entries

class config:
    def __init__(self):

        # provide system dynamics
        self.sys_dt = 0.1 # system dynamics time step [Note]: you will need to recalculate the Control invariant set if you change this value
        self.damping_factor = 0.1  ## system dynamics damping factor # [Note]: you will need to recalculate the Control invariant set if you change this value
        
        # State transition matrix with damping
        dt = self.sys_dt; damping_fun = 1 - self.damping_factor * dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, damping_fun, 0],
                           [0, 0, 0, damping_fun]])
        
        self.B = np.array([[0.5 * dt**2, 0],
                   [0, 0.5 * dt**2],
                   [dt, 0],
                   [0, dt]])


        ##################################### SIMULATION SETTINGS ##############################
        self.joystick_frequency = 20 # joystick publishing frequency
        self.controller_frequency = 200 # controller frequency
        self.simulation_frequency = 1/self.sys_dt # simulation frequency [Note]: you will need to recalculate the Control invariant set if you change this value
        self.disturbance = True

        ########################################## PATH SETTINGS ###############################

        project_root = str(Path(__file__).resolve().parent)
        print ("project_root", project_root)
        self.root_path = project_root + '/' 
        self.lib_path = self.root_path + 'lib'
        self.ros_path = self.root_path + 'ros'
        self.ros_setup_path = self.ros_path + "/devel/setup.bash"
        self.matlab_cis_path = self.lib_path + "/CIS_matlab_code/cis2m/matlab/"
        self.save_cis_path = self.lib_path + "/CIS_data/real_exp_franka_rho_tri" + f"{float(self.sys_dt):.6f}".rstrip('0').rstrip('.') + ".npz"
        self.save_cis_path = self.lib_path + "/CIS_data/real_exp_franka" + f"{float(self.sys_dt):.6f}".rstrip('0').rstrip('.') + ".npz"
        print("save_cis_path", self.save_cis_path)
        ############################################################################################

        ############################### CONSTRAINTS SETTINGS #################################
        # Model disturbances
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
     

        # provide environment
        self.agent_size = 0.019 # diameter
        scale = 0.001 #mm to m
        self.Workspace = scale*np.array([[178, 47], [704, 47], [704, 817], [178, 817]], dtype=float)
        Obs1 = scale*np.array([[568, 429], [703, 429], [703, 816], [568, 816]], dtype=float)
        Obs2 = scale*np.array([[308, 181], [438, 181], [438, 691], [308, 691]], dtype=float)
        Obs3 = scale*np.array([[437,  47], [568,  47], [568, 309], [437, 309]], dtype=float)
        Obs4_rhombus = scale * np.array([
        [640, 235],   # top
        [615, 200],   # left
        [640, 165],   # bottom
        [665, 200],   # right
        ], dtype=float)

        # Small triangle in the free top-left area
        # base ~ y=740, apex ~ y=770
        Obs5_triangle = scale * np.array([
        [230, 740],   # left base
        [260, 740],   # right base
        [245, 770],   # apex
        ], dtype=float)


        self.Obs = [Obs1, Obs2, Obs3, Obs4_rhombus, Obs5_triangle]
        self.vel_vertices   = np.array([[0.05, 0.05], [0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05]], dtype=float) #m/s
        self.accel_vertices = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]], dtype=float)         #m/s^2

        ## construct env
        helper = ConstraintHelper(eps=0.0, agent_radius=self.agent_size)
        self.Fx, self.fx, self.Fx_hs_list, self.fx_hs_list, self.Gu, self.gu, self.obs_Fx, self.obs_fx = helper.construct_state_constraints(
        workspace_vertices=self.Workspace,
        obstacle_vertices_list=self.Obs,
        vel_vertices=self.vel_vertices,
        accel_vertices=self.accel_vertices,
        )

        ## construct Control invariant sets required

        cis_exists = os.path.isfile(self.save_cis_path)
        if cis_exists:
                print("[CIS cache] Loading saved CIS from:", self.save_cis_path)
                saved_CIS = load_cis_npz(self.save_cis_path)
                self.Cx = saved_CIS["Cx"]
                self.cx = saved_CIS["cx"]
                self.Cx_dis_list = saved_CIS["Cx_dis_list"]
                self.cx_dis_list = saved_CIS["cx_dis_list"]
        else:
                print("[CIS cache] No matching file found; Computing CIS.")
                cis_builder = ExtractCIS(self)
                cis_builder.compute_cis()
                cis_builder.save_cis_npz()
                self.Cx = cis_builder.Cx; 
                self.cx = cis_builder.cx; 
                self.Cx_dis_list = cis_builder.Cx_dis_list; 
                self.cx_dis_list = cis_builder.cx_dis_list; 
        ## check if CIS if healthy
        assert_has_any_entries(self.Cx_dis_list)
        print("testing")
        ################# STATE CONVEX CONSTRAINTS ######################

        ################# CONTROL CONVEX CONSTRAINTS ####################

        ####### STATE CONVEX UNSAFE CONSTRAINTS NEED TO BE AVOIDED ######


        ################ STATE NONCONVEX CONSTRAINTS ####################
        # self.C_list =  [[np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]]),
        #         np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]])],
        #         [np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]]),
        #         np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]]),
        #         np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]]),
        #         np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]])],
        #         [np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]]),
        #         np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]]),
        #         np.array([[  99.50248756,    0.        ,    5.02487562,    0.        ],
        #                 [ -99.50248756,    0.        ,   -5.02487562,    0.        ],
        #                 [   0.        ,  140.71602592,    0.        ,    0.        ],
        #                 [-140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        , -140.71602592,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,   -7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,   -7.07106781],
        #                 [ 140.71602592,    0.        ,    0.        ,    0.        ],
        #                 [   0.        ,    0.        ,    7.07106781,    0.        ],
        #                 [   0.        ,    0.        ,    0.        ,    7.07106781],
        #                 [   0.        ,   99.50248756,    0.        ,    5.02487562],
        #                 [   0.        ,  -99.50248756,    0.        ,   -5.02487562]])]]
        
        # self.c_list = [[np.array([ 66.25496661, -58.20628929,  93.89882368, -82.23075222,
        #                         1.32257904,   0.21771409,   0.30256617,  25.42265622,
        #                         0.24670016,   0.3258322 ,  18.13674273,   1.06842661]),
        #                 np.array([ 70.58332482, -58.20628929, -13.52296169, -82.23075222,
        #                         0.21771409,   0.30256617, 100.01997081,  25.42265622,
        #                         0.24670016,   0.3258322 ,  18.13674273,  -9.42908583])],
        #                 [np.array([ 59.8868074 , -58.20628929,  84.89299802, -82.23075222,
        #                         1.32257904,   0.21771409,   0.30256617,  25.42265622,
        #                         0.24670016,   0.3258322 ,  18.13674273,   1.06842661]),
        #                 np.array([ 70.58332482, -65.51972212, -92.57338013,   1.32257904,
        #                         0.21771409,   0.30256617, 100.01997081,  25.42265622,
        #                         0.24670016,   0.3258322 ,  18.13674273,   1.06842661]),
        #                 np.array([ 70.58332482, -58.20628929,   1.71200585, -82.23075222,
        #                         1.32257904,   0.21771409,   0.30256617, 100.01997081,
        #                         0.24670016,   0.3258322 ,   1.37057357,   1.06842661]),
        #                 np.array([ 70.58332482, -58.20628929, -22.2473553 , -82.23075222,
        #                         0.21771409,   0.30256617, 100.01997081,  25.42265622,
        #                         0.24670016,   0.3258322 ,  18.13674273, -15.59824006])],
        #                 [np.array([ 63.070887  , -58.20628929,  89.39591085, -82.23075222,
        #                         1.32257904,   0.21771409,   0.30256617,  25.42265622,
        #                         0.24670016,   0.3258322 ,  18.13674273,   1.06842661]),
        #                 np.array([ 70.58332482, -68.70380172, -97.07629296,   1.32257904,
        #                         0.21771409,   0.30256617, 100.01997081,  25.42265622,
        #                         0.24670016,   0.3258322 ,  18.13674273,   1.06842661]),
        #                 np.array([ 70.58332482, -58.20628929,  14.93931229, -82.23075222,
        #                         1.32257904,   0.21771409,   0.30256617, 100.01997081,
        #                         0.24670016,   0.3258322 ,  10.72380741,   1.06842661])]]

        #######################################################################################

        ###################################### SYSTEM DYNAMICS #################################
        # self.sys_dt = 0.1 # system dynamics time step [Note]: you will need to recalculate the Control invariant set if you change this value
        # self.damping_factor = 0.1  ## system dynamics damping factor # [Note]: you will need to recalculate the Control invariant set if you change this value
        
        # # State transition matrix with damping
        # dt = self.sys_dt; damping_fun = 1 - self.damping_factor * dt
        # self.A = np.array([[1, 0, dt, 0],
        #                    [0, 1, 0, dt],
        #                    [0, 0, damping_fun, 0],
        #                    [0, 0, 0, damping_fun]])
        
        # self.B = np.array([[0.5 * dt**2, 0],
        #            [0, 0.5 * dt**2],
        #            [dt, 0],
        #            [0, dt]])
            
   ########################################################################################

        # self.consider_only_constraints = [] ## you can select constraints specified in nonconvex setting
        # map_name = 'real_world_exp_dist.png'
        # self.user_study_path = self.root_path + 'user_study'
        # self.user_study_data_path = self.root_path + 'user_study/data'
        # self.image_map_path = self.root_path + 'lib/extra/map/' + map_name
        ########################################################################################

if __name__ == "__main__":
    cfg = config()