import numpy as np
import scipy.sparse as sp
from miqp_optimizer import miqpOptimizer
import time 
import sys, os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root) +'/')
import config as config

class AssistiveControl:
    """
    One-step MIQP builder/solver that follows the MPC_CIS_testing.py block structure with sparse matrices.
    Decision vector: z = [x_k (n); x_{k+1} (n); u_k (m); p_vec (n_bin)]
    """

    def __init__(self, A, B, CFxmat, cfxvec, Gumat, guvec,
                C_list=None, c_list=None,
                 M=10e8, u_ref=None,
                 gurobi_params=None, method="cvxpy", warm_start=False, solver_verbose=False):
        """
        Args:
            A, B: system matrices
            ## not needed if
            CFxmat, cfxvec:  convex state sets, each pair (Gx_i, Fx_i) s.t. Gx_i @ x <= Fx_i
            Gumat, fuvec:  convex input sets, each pair (Gu_i, Fu_i) s.t. Gu_i @ u <= Fu_i
            C_list, c_list:   (optional) RCIS region; list(list(np.array())) [[np.array([C0,0]), np.array([C0,0]), np.array([C0,0])], ..., np.array([Ci,j])]]  Ci,j x ≤ ci,j 
            M: Big-M constant
            u_ref: references (defaults 0)
            gurobi_params: dict of Gurobi params (e.g., {"OutputFlag": 0, "TimeLimit": 1.0})
            method: solver method (e.g., "cvxpy" or "gurobi")
            warm_start: whether to use warm start (default: False)
            solver_verbose: whether to enable solver verbosity (default: False)
        """
        
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]

        self.CFxmat = CFxmat if CFxmat is not None else sp.csr_matrix((0, self.n))
        self.cfxvec = cfxvec if cfxvec is not None else np.zeros(self.CFxmat.shape[0])
        self.Gumat = Gumat if Gumat is not None else sp.csr_matrix((0, self.m))
        self.guvec = guvec if guvec is not None else np.zeros(self.Gumat.shape[0])

        self.C_list = C_list or []
        self.c_list = c_list or []

        self.M = float(M)
        # initialize input reference
        self.u_ref = np.zeros(self.m) if u_ref is None else np.asarray(u_ref, dtype=np.float64).reshape(-1)
        # Gurobi params
        self.gurobi_params = gurobi_params or {"OutputFlag": 0}
        self.x0 = np.zeros(self.n)
        
        ## Initialize binary variables
        self.n_bin = 0

        # Solver settings
        self.method = method
        self.warm_start = warm_start
        self.solver_verbose = solver_verbose
        
    def init_prob(self, x0 = None, u_ref = None):
        if x0 is not None:
            self.x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if u_ref is not None:
            self.u_ref = np.asarray(u_ref, dtype=np.float64).reshape(-1)

        self.construct_sys_dyn()
        self.construct_state_convex_inequalities()
        self.construct_state_nonconvex_inequalities()
        self.construct_control_convex_inequalities()
        self.build_constraints()
        self.build_objective()
        self.miqp_optimizer = miqpOptimizer(self.Qcal, self.qvec, self.Acal,
                                             self.z_lb, self.z_ub, self.n, 
                                             self.m, method=self.method, 
                                             warm_start=self.warm_start, verbose=self.solver_verbose)

    def update_prob(self, x0=None, u_ref=None):
        if x0 is not None:
            self.update_bounds(x0)

        if u_ref is not None:
            self.update_objective(u_ref)

        self.miqp_optimizer.update(self.qvec, self.z_lb, self.z_ub)

    def solve_prob(self):
        res = self.miqp_optimizer.solve()

        if res['status'] != 1:
            print("Warning: solver status is", res['status'])
            self.xnext = None
            self.u_asst = None
            self.p_bin = None
            self.feasible = 0
        else:    
            self.xnext = res['z'][self.n:self.n*2].ravel()
            self.u_asst = res['z'][self.n*2:self.n*2+self.m].ravel()
            self.p_bin = res['z'][self.n*2+self.m:].ravel()
            self.feasible = 1

        ## To Do: implement containment test: if check containment else: optimize
        return res
            
    def construct_sys_dyn(self):
        n, m = self.n, self.m

        ## Need to update in the paper:
        A_ = np.block([
            [np.zeros((n, n)), np.zeros((n, n))],
            [self.A, np.zeros((n, n))]
        ])
        I_2n = sp.eye(2 * n)
        self.A_prime = A_ - I_2n


        self.B_prime = np.vstack([
            np.zeros((n, m)),
            self.B
        ])

        # Final equality constraint matrix
        self.A_eq_dyn = sp.csr_matrix(np.hstack([
            self.A_prime,
            self.B_prime
        ]))



    def construct_state_convex_inequalities(self):
        # Extend Fmat to act on xvec_k = { [x_k, x_{k+1}] ∈ R^{2n} | [0 | F][x_k, x_{k+1}].T <= f }
        self.Fcal = sp.hstack([      
                              sp.csr_matrix((self.CFxmat.shape[0], self.n)),   # zeros but sparse
                              self.CFxmat              # ensure CFxmat is sparse
                              ], format="csr")                           
        self.fvec = self.cfxvec


    def construct_control_convex_inequalities(self):
        # Extend Gmat to act on {u_k ∈ R^{m} | G u_k <= g }
        self.Gcal = sp.csr_matrix(self.Gumat)
        self.gvec = self.guvec

    def construct_state_nonconvex_inequalities(self):
        C_list = self.C_list; c_list = self.c_list
        n = self.n

        Ccal_blocks = []
        cvec_ub     = []
        L_blocks    = []

        E_blocks    = []
        Epvec_lb = []
        Epvec_ub = []

        for i, (C_i, c_i) in enumerate(zip(C_list, c_list)):
            # ----- MCIS inequality pieces -----
            for j, (C_ij, c_ij) in enumerate(zip(C_i, c_i)):
                C_ij = sp.csr_matrix(C_ij) if not sp.issparse(C_ij) else C_ij
                c_ij = np.asarray(c_ij).ravel()
                mC_ij = C_ij.shape[0]
                print ("i, j", i, j)
                print ("sp.csr_matrix((mC_ij, n)),", sp.csr_matrix((mC_ij, n)).shape)
                # [0_{mC_ij x n}, C_ij] applied to [x_k; x_{k+1}]
                Ccal_ij = sp.hstack([sp.csr_matrix((mC_ij, n)), C_ij], format="csr")
                Ccal_blocks.append(Ccal_ij)
                cvec_ub.append(c_ij)

                L_pij = sp.csr_matrix(np.ones((mC_ij, 1)))
                L_blocks.append(L_pij)

            # ----- Binary aggregation constraints for this region (Eq. 13) -----
            nCi = len(C_i)  # number of (i,j) constraints in region i
            oneNci = sp.csr_matrix(np.ones((1, nCi)))

            # E_i = 1^T_{nCi} X I_2  -> shape (2, 2*nCi)
            E_blocks.append(oneNci)

            # bounds: p_i_lower = [0,0], p_i_upper = (nCi-1)*[1,1]
            Epvec_lb.append([0])
            Epvec_ub.append([nCi - 1])


        # ----- Final matrices -----
        self.Ccal = sp.vstack(Ccal_blocks, format="csr") if Ccal_blocks else sp.csr_matrix((0, 2*n))
        self.cvec = np.concatenate(cvec_ub) if cvec_ub else np.array([])

        self.Lp = sp.block_diag(L_blocks, format="csr") if L_blocks else sp.csr_matrix((0, 0))

        # E, E_lb, E_ub for all regions (block-diagonal over Ei)
        self.E      = sp.block_diag(E_blocks, format="csr") if E_blocks else sp.csr_matrix((0, 0))
        self.E_lb   = np.concatenate(Epvec_lb, dtype=np.int32) if Epvec_lb else np.array([], dtype=np.int32)
        self.E_ub   = np.concatenate(Epvec_ub, dtype=np.int32) if Epvec_ub else np.array([], dtype=np.int32)
        self.n_bin = self.E.shape[1]  # total number of binary variables
        # print("Number of binary variables:", self.n_bin)

    def update_bounds(self, x0):
        self.X0 = np.concatenate([-x0, np.zeros(self.n)])
        # Update the bounds for the optimization problem
        self.z_ub[:self.X0.shape[0]] = self.X0
        self.z_lb[:self.X0.shape[0]] = self.X0

    def build_constraints(self):
        # Combine all constraints into a single sparse matrix
        
        # first column
        Ar4c1 = sp.csr_matrix((self.Gcal.shape[0], self.Ccal.shape[1]))
        Ar5c1 = sp.csr_matrix((self.E.shape[0], self.Ccal.shape[1]))
        A_col1 = sp.vstack([self.A_prime, self.Fcal, self.Ccal, Ar4c1, Ar5c1], format="csr")

        # second column
        Ar2c2 = sp.csr_matrix((self.Fcal.shape[0], self.Gcal.shape[1]))
        Ar3c2 = sp.csr_matrix((self.Ccal.shape[0], self.Gcal.shape[1]))
        Ar5c2 = sp.csr_matrix((self.E.shape[0], self.Gcal.shape[1]))
        A_col2 = sp.vstack([self.B_prime, Ar2c2, Ar3c2, self.Gcal, Ar5c2], format="csr")

        # third column
        Ar1c3 = sp.csr_matrix((self.A_prime.shape[0], self.E.shape[1]))
        Ar2c3 = sp.csr_matrix((self.Fcal.shape[0], self.E.shape[1]))
        Ar4c3 = sp.csr_matrix((self.Gcal.shape[0], self.E.shape[1]))
        A_col3 = sp.vstack([Ar1c3, Ar2c3, -self.M*self.Lp , Ar4c3, self.E], format="csr")
        self.Acal = sp.hstack([A_col1, A_col2, A_col3], format="csr")
        
        self.X0 = np.concatenate([-self.x0, np.zeros(self.n)])

        self.z_ub = np.concatenate([self.X0, self.fvec, self.cvec, self.gvec, self.E_ub])
        z_lb_2 = -np.inf*np.ones(self.Fcal.shape[0])
        z_lb_3 = -np.inf*np.ones(self.Ccal.shape[0])
        z_lb_4 = -np.inf*np.ones(self.Gcal.shape[0])
        self.z_lb = np.concatenate([self.X0, z_lb_2, z_lb_3, z_lb_4, self.E_lb])

        return self.Acal, self.z_lb, self.z_ub

    def update_objective(self, uref):
        self.u_ref = uref
        qu = (-self.u_ref.T@self.Qucal).reshape(-1, 1)
        t0 = time.perf_counter()
        self.qvec = sp.vstack([self.qx, qu, self.qp], format="csr")

    def build_objective(self):
        Qxcal = sp.csr_matrix((self.n*2, self.n*2))
        self.Qucal = sp.eye(self.m, format="csr")
        Qpcal = sp.csr_matrix((self.n_bin, self.n_bin))
        self.Qcal = sp.block_diag([Qxcal, self.Qucal, Qpcal], format="csr")

        self.qx = sp.csr_matrix((self.n*2, 1))
        self.qu = (-self.u_ref.T@self.Qucal).reshape(-1, 1)
        self.qp = 0*sp.csr_matrix((self.n_bin, 1))
        self.qvec = sp.vstack([self.qx, self.qu, self.qp], format="csr")


if __name__ == "__main__":
    cfg = config.config()
    import matplotlib.pyplot as plt
    from visualization import Plotter
    from helper import load_cis_npz

    saved_CIS = load_cis_npz(cfg.save_cis_path)

    A = saved_CIS["A"]
    B = saved_CIS["B"]
    Cx = saved_CIS["Cx"]
    cx = saved_CIS["cx"]
    Cx_dis_list = saved_CIS["Cx_dis_list"]
    cx_dis_list = saved_CIS["cx_dis_list"]
    Fx_hs_list = saved_CIS["Fx_hs_list"]
    fx_hs_list = saved_CIS["fx_hs_list"]
    Gu = saved_CIS["Gu"]
    gu = saved_CIS["gu"]
    E = saved_CIS["E"]
    Gw = saved_CIS["Gw"]
    Fw = saved_CIS["Fw"]
    fw = saved_CIS["fw"]
    Fx = saved_CIS["Fx"]
    fx = saved_CIS["fx"]
    
    # concatenate workspace and obstacle for plotting
    obs_Fx = cfg.obs_Fx.copy()
    obs_Fx.append(Fx)
    obs_fx = cfg.obs_fx.copy()
    obs_fx.append(fx)

    plotter = Plotter(cfg)

    # C_list = saved_CIS["C_list"]
    # c_list = saved_CIS["c_list"]


    N = 100
    xref = []
    uref = [[0.1, -1] for i in range(N-1)]
    # F = cfg.Fx
    # f = cfg.fx
    F = None; f = None
    # A = cfg.A; B = cfg.B
    G = None; g = None
    # C_list = cfg.C_list; c_list = cfg.c_list

    miqp = AssistiveControl(A, B, CFxmat = Cx, cfxvec = cx, Gumat = G, guvec = g,
                    C_list=Cx_dis_list, c_list=cx_dis_list,
                    M=10e8, u_ref=None,
                    gurobi_params=None, method = "gurobi", warm_start=False)

    x0 = np.array([0.695, 0.160, 0.055, 0.055])
    miqp.init_prob(x0)

    N = 2
    uref = np.vstack([np.ones((1, N)), np.zeros((1, N))]).T

    print ("uref.shape", uref.shape)
    print ("uref[0]", uref[1].shape)
    miqp.update_prob(x0, uref[0])
    miqp.solve_prob()

    xnext = miqp.xnext
    u_asst = miqp.u_asst

    print("A:", miqp.A.shape)
    print("B:", miqp.B.shape)
    print("Q:", miqp.Qcal.shape)
    print("q:", miqp.qvec.shape)
    print("Acal:", miqp.Acal.shape)
    print("z_lb:", miqp.z_lb.shape)
    print("z_ub:", miqp.z_ub.shape)
    print("Ep:", miqp.E.shape)
    print("n_bin:", miqp.n_bin)
    print("")

    ### simple example: 
    E = cfg.E
    Gw = cfg.Gw
    Fw = cfg.Fw

    nx = A.shape[0]
    nu = B.shape[1]
    rng = np.random.default_rng(42)           
    nw = E.shape[1]
    ub = Fw[:nw].reshape(nw, 1)
    lb = -Fw[nw:].reshape(nw, 1)
    
    N = 100
    uref = np.vstack([np.ones((1, N)), np.zeros((1, N))]).T
    print(uref)
    x0 = np.array([0.695, 0.160, 0.045, 0.045])

    x_hist = [x0]
    u_hist = []
    feasible_list = []
    import time
    plt.ion()
    for i in range(N):
        print ("iteration", i)
        ### MPC 
        t0 = time.perf_counter()
        # miqp.init_prob(x0) # note required
        miqp.update_prob(x0, uref[0])
        miqp.solve_prob()
        t1 = time.perf_counter()
        print("MPC solve time:", t1 - t0, "Freq:", 1/(t1 - t0))
        feasible_list.append(miqp.feasible)
        print ("number of times feasible:", sum(feasible_list), "out of", N)
        ####
        
        u_asst = miqp.u_asst
        print ("u_asst", u_asst)
        
        if u_asst is None:
            # infeasible -> use zero input (or keep last)
            u_asst = np.zeros(nu)
        u_hist.append(u_asst.copy())

        # Sample uniform disturbance in [lb, ub] per component
        w = lb + (ub - lb) * rng.random((nw, 1))
        w = (w).reshape(nw,)   # optional scaling like your example

        x0 = A @ x0 + B @ u_asst + E @ w
        print("next_x", x0)
        x_hist.append(x0)
        u_hist.append(u_asst)

        # visualization
        # plotter.set_point(x0)
        vel = np.array(x_hist)[:, 2:4]
        len_vel = vel.shape[0]
        t = np.arange(0.1, len_vel*0.1+0.001, 0.1)
        # print (t)
        # print (vel)
        # plotter.set_velocity(vel, xlim=0.1*i)
        # time.sleep(0.1)
        plt.pause(0.1)

    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)

    # print("No. of time solution were not feasible:", sum(feasible_list) - len(feasible_list))
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_hist[:, 0], x_hist[:, 1])
    plt.title("State History")
    plt.subplot(2, 1, 2)
    plt.plot(u_hist[:, 0])
    plt.plot(u_hist[:, 1])
    plt.title("Control History")
    plt.show()