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
    def __init__(self, cfg):
        # Needed for paths
        self.cfg = cfg

        # Define the system dynamics
        self.A = cfg.A
        self.B = cfg.B

        # Define state and input constraints
        # Nonconvex state safe set supported by halfspaces
        self.Fx_hs_list = cfg.Fx_hs_list  # decompose halfspaces for non-convex safe sets
        self.fx_hs_list = cfg.fx_hs_list  # decompose halfspaces for non-convex safe sets
        
        # convex state safe set
        self.Fx = cfg.Fx if isinstance(cfg.Fx, np.ndarray) else None
        self.fx = cfg.fx if isinstance(cfg.fx, np.ndarray) else None

        self.Gu = cfg.Gu if isinstance(cfg.Gu, np.ndarray) else None
        self.gu = cfg.gu if isinstance(cfg.gu, np.ndarray) else None

        # Define disturbance constraints
        self.E = cfg.E
        self.Gw = cfg.Gw
        self.Fw = cfg.Fw

        self.cis_method = False # for MPT3 matlab function 
        self.L = 2
        self.T = 2

        self.initialize_inequality()

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
        
    @staticmethod
    def _to_ml_mat(X):
        X = np.asarray(X, dtype=float)
        return matlab.double(X.tolist())

    @staticmethod
    def _to_ml_col(x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return matlab.double(x.tolist())

    @staticmethod
    def _cell_to_py(cell):
        # MATLAB cell array (of matrices/vectors) -> list of np.ndarrays
        return [np.array(c, dtype=float) for c in cell]

    def initialize_inequality(self):
        # constant matrices to MATLAB types (ensure column vectors for *_w, *_u)
        self.A_matlab  = self._to_ml_mat(self.A)
        self.B_matlab  = self._to_ml_mat(self.B)

        self.Gu_matlab = self._to_ml_mat(self.Gu) if self.Gu is not None else matlab.double([])
        self.gu_matlab = self._to_ml_col(self.gu) if self.gu is not None else matlab.double([])

        self.E_matlab  = self._to_ml_mat(self.E)  if self.E  is not None else matlab.double([])
        self.Gw_matlab = self._to_ml_mat(self.Gw) if self.Gw is not None else matlab.double([])
        self.Fw_matlab = self._to_ml_col(self.Fw) if self.Fw is not None else matlab.double([])

    def compute_cis(self):
        self.cis_data = []

        # Start MATLAB engines
        num_jobs = sum(len(block) for block in self.Fx_hs_list)
        num_engines = max(1, min(10, max(1, len(self.Fx_hs_list) * 4), num_jobs))
        print("<<<<<<<<<< matlab engine initialization >>>>>>>>>")
        # t0 = time.perf_counter()
        engines = [self.start_matlab_engine(self.cfg.lib_path)
                for _ in range(num_engines)]
        # print("time taken for matlab engine initialization:", time.perf_counter() - t0)

        # Constants (already prepared in initialize_inequality)
        A_ml, B_ml = self.A_matlab, self.B_matlab
        Gu_ml, gu_ml = self.Gu_matlab, self.gu_matlab
        E_ml, Gw_ml, Fw_ml = self.E_matlab, self.Gw_matlab, self.Fw_matlab
        implicit_ml = matlab.logical(bool(self.cis_method))
        L = float(self.L) if self.L is not None else 0.0
        has_T = self.T is not None
        T = float(self.T) if has_T else 0.0

        ### compute safe-sets for non-convex region using admissible sets (x \in X, u \in U) and non-convex decomposed into halfspaces
        # Launch jobs (round-robin over engines)
        job_list = []  # (i_obs, i_facet, future)
        eng_idx = 0
        if self.Fx_hs_list is not None and self.fx_hs_list is not None:
            for i_obs, (Fx_hs_block, fx_hs_block) in enumerate(zip(self.Fx_hs_list, self.fx_hs_list)):
                jobs = []
                for i_facet, (F_i, f_i) in enumerate(zip(Fx_hs_block, fx_hs_block)):
                    if F_i.shape[1] != self.A.shape[0]:
                        raise ValueError(
                            f"Fx[{i_obs}][{i_facet}] has {F_i.shape[1]} cols but A has {self.A.shape[0]} rows."
                        )
                    eng = engines[eng_idx % num_engines]
                    eng_idx += 1

                    Fx_hs_ml = matlab.double(np.asarray(F_i, dtype=float).tolist())
                    fx_hs_ml = matlab.double(np.asarray(f_i, dtype=float).reshape(-1, 1).tolist())

                    if has_T:
                        fut = eng.computeRCIS_extract(
                            A_ml, B_ml, Fx_hs_ml, fx_hs_ml, Gu_ml, gu_ml,
                            E_ml, Gw_ml, Fw_ml, implicit_ml, L, T,
                            nargout=3, background=True
                        )
                    else:
                        fut = eng.computeRCIS_extract(
                            A_ml, B_ml, Fx_hs_ml, fx_hs_ml, Gu_ml, gu_ml,
                            E_ml, Gw_ml, Fw_ml, implicit_ml, L,
                            nargout=3, background=True
                        )
                    jobs.append((i_obs, i_facet, fut))
                job_list.append(jobs)

            # Collect
            Cx_dis_list = []
            cx_dis_list = []

            for i, jobs in tqdm(enumerate(job_list)):
                C_ = []
                c_ = []
                for j, (i_obs, i_facet, fut) in tqdm(enumerate(jobs)):
                    H_cell, f_cell, volume = fut.result()  # numeric-only types ✔
                    # Convert MATLAB cell -> Python lists of numpy arrays
                    # def _cell_to_np_list(cell): return [np.array(c, dtype=float) for c in cell]
                    H_struct = np.array(H_cell[0], dtype=float)
                    f_struct = np.array(f_cell[0], dtype=float).squeeze()
                    if H_struct.size != 0:
                        C_.append(H_struct)
                        c_.append(f_struct)
                Cx_dis_list.append(C_)
                cx_dis_list.append(c_)
                
            self.Cx_dis_list = Cx_dis_list
            self.cx_dis_list = cx_dis_list
        else:
            self.Cx_dis_list = None
            self.cx_dis_list = None

        # compute safe-sets for convex region using admissible sets (x \in X, u \in U)
        if isinstance(self.Fx, np.ndarray) and isinstance(self.fx, np.ndarray):
            Fx_ml, fx_ml = self._to_ml_mat(self.Fx), self._to_ml_col(self.fx)
            res = eng.computeRCIS_extract(
                A_ml, B_ml, Fx_ml, fx_ml, Gu_ml, gu_ml,
                E_ml, Gw_ml, Fw_ml, implicit_ml, L,
                nargout=3, background=True
            )

            H_cell, f_cell, volume = res.result()
            self.Cx = np.array(H_cell[0], dtype=float)
            self.cx = np.array(f_cell[0], dtype=float).squeeze()
        else:
            self.Cx = None
            self.cx = None

        for eng in engines:
            try: eng.quit()
            except Exception: pass


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
    def start_matlab_engine(matlab_cis_path):
        #matlab engine 
        eng = matlab.engine.start_matlab()

        print ("matlab engine started: path", matlab_cis_path)
        ExtractCIS.add_path(eng, matlab_cis_path)
        return eng

    @staticmethod
    def quit_matlab(eng):
        eng.quit()

    def save_cis_npz(self):
        
        out_path = self.cfg.save_cis_path

        # Ensure path & extension
        out_path = os.fspath(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if not out_path.lower().endswith(".npz"):
            out_path += ".npz"

        # Pack data (object arrays for nested lists)
        pack = {
            # Computed convex state safe set CIS constraint
            "Cx": np.array(self.Cx, dtype=float) if self.Cx is not None else None,
            "cx": np.array(self.cx, dtype=float).squeeze() if self.cx is not None else None,

            # Computed disjunctive state safe set CIS constraints 
            "Cx_dis_list":   np.array(self.Cx_dis_list, dtype=object) if self.Cx_dis_list is not None else None,
            "cx_dis_list":   np.array(self.cx_dis_list, dtype=object) if self.cx_dis_list is not None else None,

            # system
            "A":  np.asarray(self.A, dtype=float),
            "B":  np.asarray(self.B, dtype=float),

            # original convex state constraints used as input to CIS
            "Fx": np.asarray(self.Fx, dtype=float) if self.Fx is not None else None,
            "fx": np.asarray(self.fx, dtype=float).squeeze() if self.fx is not None else None,

            # original disjunctive state constraints used as input to CIS
            "Fx_hs_list": np.array(self.Fx_hs_list, dtype=object) if self.Fx_hs_list is not None else None,
            "fx_hs_list": np.array(self.fx_hs_list, dtype=object) if self.fx_hs_list is not None else None,

            # input constraints
            "Gu": np.asarray(self.Gu, dtype=float) if self.Gu is not None else None,
            "gu": np.asarray(self.gu, dtype=float).squeeze() if self.gu is not None else None,

            # disturbance mapping & set
            "E":  np.asarray(self.E,  dtype=float) if self.E  is not None else None,
            "Gw": np.asarray(self.Gw, dtype=float) if self.Gw is not None else None,
            "Fw": np.asarray(self.Fw, dtype=float).squeeze() if self.Fw is not None else None,
            # alias for user’s naming (sometimes they refer to 'fw')
            "fw": np.asarray(self.Fw, dtype=float).squeeze() if self.Fw is not None else None,

            # CIS config
            "L": float(self.L) if getattr(self, "L", None) is not None else np.nan,
            "T": float(self.T) if getattr(self, "T", None) is not None else np.nan,
            "implicit": bool(getattr(self, "cis_method", False)),
        }

        # Save compressed .npz
        np.savez_compressed(out_path, **pack)
        print(f"[CIS] Saved to: {out_path}")
        return out_path

if __name__ == "__main__":

    import sys, os
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    
    sys.path.insert(0, str(project_root) +'/')
    from config import config
    cfg = config()

    from constraint_builder import ConstraintHelper

    helper = ConstraintHelper(eps=0.0)

    Fx, fx, Fx_hs_list, fx_hs_list, Gu, gu = helper.construct_state_constraints(
        workspace_vertices=cfg.Workspace,
        obstacle_vertices_list=cfg.Obs,
        vel_vertices=cfg.vel_vertices,
        accel_vertices=cfg.accel_vertices,
    )
    cfg.Fx = Fx
    cfg.fx = fx
    cfg.Fx_hs_list = Fx_hs_list
    cfg.fx_hs_list = fx_hs_list
    cfg.Gu = Gu
    cfg.gu = gu

    from CIS_computation import ExtractCIS
    cis_builder = ExtractCIS(cfg)

    cis_builder.compute_cis()
    cis_builder.save_cis_npz()
    # print("CIS computation finished.")
    # print("Cx_dis_list:", cis_builder.Cx_dis_list)
    # print("cx_dis_list:", cis_builder.cx_dis_list)
