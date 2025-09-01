# https://github.com/cyphylab/cis-supervisor/blob/main/matlab/quadrocopter_supervision/scenario_setup.m

from array import array
import numpy as np           
import sys
import os
import psutil
import matlab.engine
import scipy.io as sio
from tqdm import tqdm
import time

class ExtractCIS:
    '''Class to extract the Control Invariant Set (CIS) from the given system dynamics'''
    def __init__(self, cfg, Fx_list = None, fx_list = None,
                 Gu = None, gu = None, E = None, Gw = None, 
                 Fw = None, use_matlab = False, safe_set_path = None):
        
        # Define the system dynamics
        self.A = cfg.A
        self.B = cfg.B

        # Define state and input constraints
        self.Fx_list = Fx_list
        self.fx_list = fx_list
        self.Gu = Gu
        self.gu = gu

        # Define disturbance constraints
        self.E = E
        self.Gw = Gw
        self.Fw = Fw

        self.cis_method = False # for MPT3 matlab function 
        self.L = 2
        self.T = 2

        self.initialize_inequality()

        self.safe_set_path = safe_set_path
        self.use_matlab = use_matlab
        self.matlab_cis_path = cfg.matlab_cis_path
        self.python_to_matlab_function_path = cfg.lib_path
        # self.kill_matlab()

    @staticmethod
    def convert_matlab_data(data):
        '''Convert MATLAB data to numpy array'''
        if isinstance(data, float):
            return data
        else:
            python_data = {}
            for i in range(len(data)):
                python_data[i] = np.array(data[i])
            return python_data
        
    def initialize_inequality(self):

        # Precompute constant matrices (no need to convert them in each loop)
        self.A_matlab = matlab.double(self.A.tolist())
        self.B_matlab = matlab.double(self.B.tolist())
        self.Gu_matlab = matlab.double(self.Gu.tolist()) if self.Gu is not None else matlab.double([])
        self.gu_matlab = matlab.double(self.gu.tolist()) if self.gu is not None else matlab.double([])


        self.E_matlab = matlab.double(self.E.tolist()) if self.E is not None else matlab.double([])
        self.Gw_matlab = matlab.double(self.Gw.tolist()) if self.Gw is not None else matlab.double([])
        self.Fw_matlab = matlab.double(self.Fw.tolist()) if self.Fw is not None else matlab.double([])

    def compute_cis(self):
        self.cis_data = []
        solution_list = []

        # Start up MATLAB engines
        num_engines = min(10, len(self.Fx_list) * 4)  # Ensure at most 10 engines, or enough for tasks
        print ("<<<<<<<<<< matlab engine initialization >>>>>>>>>: ")
        t0 = time.perf_counter()
        ## CIS settings:
        L = self.L
        T = self.T
        implicit = False
        implicit_matlab = matlab.logical(implicit)
        cis_method = matlab.logical(self.cis_method)

        print ("Computing CIS for INNER BOX")
        
        engines = [self.start_matlab_engine(self.matlab_cis_path, self.python_to_matlab_function_path) for _ in range(num_engines)]
        print ("time taken for matlab engine initialization: ", time.perf_counter() - t0)
        # Distribute tasks across MATLAB engines in round-robin
        engine_idx = 0
        for Fx_, fx_ in zip(self.Fx_list, self.fx_list):
            solutions = []
            for (Fx, fx) in zip(Fx_, fx_):
                if Fx.shape[1] != self.A.shape[0]:
                    raise ValueError(f"Dimension mismatch: Fx has {Fx.shape[1]} columns but A has {self.A.shape[0]} rows.")
                
                # Launch asynchronous MATLAB computations for each condition using available engines
                eng = engines[engine_idx % num_engines]  # Rotate among engines
                engine_idx += 1
                Gx_matlab = matlab.double(Fx.tolist())
                bx_matlab = matlab.double(fx.tolist())

                # Run MATLAB function asynchronously and store the future
                future = eng.computeRCIS_py(
                    self.A_matlab, self.B_matlab, Gx_matlab, bx_matlab,
                    self.Gu_matlab, self.gu_matlab, self.E_matlab, self.Gw_matlab, self.Fw_matlab, implicit_matlab, L, T, self.matlab_cis_path , cis_method, nargout=3, background=True
                )
                solutions.append(future)

            solution_list.append(solutions)

        # Collect results once all computations are started
        C_list = []
        c_list = []
        for solution_list_ in tqdm(solution_list):
            C_list_ = []
            c_list_ = []
            for solution_ in solution_list_:
                print("\n <<< Box number: >>>", idx)
                [H_struct, f_struct, volume] = solution_.result()  # Retrieve MATLAB result
                H_struct = ExtractCIS.convert_matlab_data(H_struct)
                f_struct = ExtractCIS.convert_matlab_data(f_struct)
                volume = ExtractCIS.convert_matlab_data(volume)
                C_list_.append(H_struct)
                c_list_.append(f_struct)
            C_list.append(C_list_)
            c_list.append(c_list_)

        self.C_list = C_list
        self.c_list = c_list

        # print ("Computing CIS for OUTER BOX")
        # # compute CIS only for convex set, i.e only for outer box
        # Gx_matlab = matlab.double(self.Gx_inc.tolist())
        # bx_matlab = matlab.double(self.bx_inc.reshape(-1, 1).tolist()) 
        # eng = self.start_matlab_engine(self.matlab_cis_path, self.python_to_matlab_function_path)
        # engines.append(eng)
        # # Call the MATLAB function computeRCIS
        # [H_struct, f_struct, volume] = eng.computeRCIS_py(
        #     A_matlab, B_matlab, Gx_matlab, bx_matlab, 
        #     Gu_matlab, bu_matlab,  E_matlab, Gw_matlab, Fw_matlab, implicit_matlab, L, T, self.matlab_cis_path , cis_method, nargout=3)  

        # H_struct = ExtractCIS.convert_matlab_data(H_struct)
        # f_struct = ExtractCIS.convert_matlab_data(f_struct)
        # volume = ExtractCIS.convert_matlab_data(volume)
        
        # self.cis_data.append({})
        # # Check if it's initialized and then add data
        # # if 'H_struct' not in self.cis_data[-1]:
        # self.cis_data[-1]['H_struct'] = H_struct
        # # if 'f_struct' not in self.cis_data[-1]:
        # self.cis_data[-1]['f_struct'] = f_struct
        # # if 'volume' not in self.cis_data[-1]:
        # self.cis_data[-1]['volume'] = volume

        # self.save_data()

        # Close all MATLAB engines
        for eng in engines:
            eng.quit()
            
    def save_data(self):
        filename = self.safe_set_path
        np.save(filename, self.cis_data)

        cis_data_matlab = self.convert_numpy_data(self.cis_data)
        # # Save the list of structs as 'cis_data' in MATLAB format
        filename = ('/').join(filename.split('.')[:-1])
        sio.savemat(filename + '.mat', {'cis_data': cis_data_matlab})

    def convert_numpy_data(self, data):

        
        last_idx = len(data) - 1
        cis_data_matlab = []
        for k, data_ in enumerate(data):
            cis_data_matlab_ = []
            # Create a list to store structs (to mimic MATLAB struct arrays)
            if last_idx == k:
                # Handle the last index with H_struct, f_struct, and volume
                cis_data_matlab_.append({
                    'H_struct': np.array(data_['H_struct'].items()),
                    'f_struct': np.array(data_['f_struct'].items()),
                    'volume': np.array(data_['volume'])
                })
            else:
                # Loop through the Python dictionary and create a MATLAB struct array
                for i, key in enumerate(data_):
                    # Check if the current data is the last index with different structure
                    # Al and Au are lists of matrices of different sizes
                    Al_list = {'key_'+str(k+1): np.array(mat) for k, mat in data_[key]['Al'].items()}  # Al as a list of NumPy arrays
                    Au_list = {'key_'+str(k+1): np.array(mat) for k, mat in data_[key]['Au'].items()}  # Au as a list of NumPy arrays

                    # bl, bu, volume_l, volume_u are dictionaries with indices, so we convert them to structs
                    bl_struct = {'key_'+str(k+1): np.array(v) for k, v in data_[key]['bl'].items()}  # Convert bl to struct-like
                    bu_struct = {'key_'+str(k+1): np.array(v) for k, v in data_[key]['bu'].items()}  # Convert bu to struct-like
                    volume_l_struct = {'volume_l': data_[key]['volume_l']}  # Convert volume_l to struct-like
                    volume_u_struct = {'volume_u': data_[key]['volume_u']}  # Convert volume_u to struct-like

                    # Append the struct-like dictionary to the list
                    cis_data_matlab_.append({
                        'Al': Al_list,  # Store list of matrices for Al
                        'bl': bl_struct,  # Store bl as a struct-like dictionary
                        'Au': Au_list,  # Store list of matrices for Au
                        'bu': bu_struct,  # Store bu as a struct-like dictionary
                        'volume_l': volume_l_struct,  # Store volume_l as a struct-like dictionary
                        'volume_u': volume_u_struct  # Store volume_u as a struct-like dictionary
                    })
            cis_data_matlab.append(cis_data_matlab_)

        return cis_data_matlab
    
    @staticmethod
    def add_path( eng, path):
        generated_path = eng.genpath(path)
        # print("Return command add path:", generated_path)

        # Add the path to MATLAB's search path
        eng.addpath(generated_path)

    @staticmethod
    def kill_matlab():
        allMatlabIds = [p.pid for p in psutil.process_iter() if "matlab" in str(p.name)]
        print ("allMatlabIds", allMatlabIds)
        MatlabIdsToKill = [x for x in allMatlabIds if x != os.getpid()]
        for MatlabId in MatlabIdsToKill:
            os.kill(MatlabId, signal.SIGINT)

    @staticmethod
    def start_matlab_engine(matlab_cis_path, python_to_matlab_function_path):
        # print ("psutil.process_iter()", psutil.process_iter())

        #matlab engine 
        eng = matlab.engine.start_matlab()
        
        # Define the path to the directory containing the MATLAB function
        # matlab_cis_path = "/home/shivam/Dropbox (Aalto)/research_work/codes/assistive_system/dev/github_code/cis2m/matlab/"
        ExtractCIS.add_path(eng, matlab_cis_path)

        # python_to_matlab_function_path = "/home/shivam/Dropbox (Aalto)/research_work/codes/assistive_system/dev/realtime_MPC/lib/"
        ExtractCIS.add_path(eng, python_to_matlab_function_path)

        # # Verify if the path exists in MATLAB’s search path
        # current_path = eng.path()
        # if matlab_function_path in current_path:
        #     print("Path added successfully")
        # else:
        #     print("Path not added")
        return eng

    @staticmethod
    def quit_matlab(eng):
        eng.quit()

def extract_cis(variables, cfg):
    use_matlab = True
    print ("extracting CIS")
    extract_cis = ExtractCIS(variables, cfg, use_matlab = use_matlab)
    extract_cis.compute_inequality()
    extract_cis.compute_cis()

if __name__ == "__main__":

    import sys, os
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    
    sys.path.insert(0, str(project_root) +'/')
    from config1 import config
    cfg = config()

    from helper import ConstraintHelper

    helper = ConstraintHelper(eps=0.0)

    Workspace = np.array([[178, 47], [704, 47], [704, 817], [178, 817]], dtype=float)
    Obs1 = np.array([[568, 429], [703, 429], [703, 816], [568, 816]], dtype=float)
    Obs2 = np.array([[308, 181], [438, 181], [438, 691], [308, 691]], dtype=float)
    Obs3 = np.array([[437,  47], [568,  47], [568, 309], [437, 309]], dtype=float)
    vel_vertices   = np.array([[0.05, 0.05], [0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05]], dtype=float)
    accel_vertices = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]], dtype=float)

    Fx_list, fx_list, Gu, gu = helper.construct_state_constraints(
        workspace_vertices=Workspace,
        obstacle_vertices_list=[Obs1, Obs2, Obs3],
        vel_vertices=vel_vertices,
        accel_vertices=accel_vertices,
    )
    # print("Fx_list:", Fx_list)
    # print("fx_list:", fx_list)
    # print("Gu:", Gu)
    # print("gu:", gu)

    E = cfg.E
    Gw = cfg.Gw
    Fw = cfg.Fw

    from CIS_computation import ExtractCIS
    cis_builder = ExtractCIS(cfg, Fx_list = Fx_list, fx_list = fx_list,
                    Gu = Gu, gu = gu, E = E, Gw = Gw, 
                    Fw = Fw, use_matlab = False, safe_set_path = None)

    cis_builder.compute_cis()