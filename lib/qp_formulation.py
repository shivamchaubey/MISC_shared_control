from scipy import sparse
import numpy as np
from numpy import hstack, inf, ones
from math import cos, sin, atan, tan, atan2, pi
from time import time, sleep
import osqp
import sys
# if scipy is used for optimization
import scipy.optimize as opt
from scipy.optimize import Bounds
# import cvxpy
import sys
# import miosqp
# import mathprogbasepy as mpbpy
import gurobipy as gp
from gurobipy import GRB

# np.set_printoptions(threshold=sys.maxsize, suppress=True, formatter={'float': '{: 0.0f}'.format}, linewidth=100000)

class QPFormulation():
    """
    Formulate MPC problem in terms of QP problem using sparse matrices

    Attributes:
        -> Settings for MPC:
            1. Apply penalty to go closer to bound  1/0
            2. Apply penalty on input rate 1/0
            3. Apply terminal safe set constraints 1/0

        -> Bounds on states, control input and control input rate.
        
        -> Weight for objective function Ju = JQx + JQu + JQdu + JQe 
            where,
            JQx  : Cost of states 
            JQu  : Cost of input
            JQdu : Cost of input rate
            JQe  : Cost to be closer to bound  
        
        -> Controller increment rate 
        -> Horizon
        -> size of states and input
        -> System dynamics matrices A and B
    solve: given current states and previous applied action computes the control action
    """

    def __init__(self, obj):


        self.opt_method_type = obj.opt_method_type
        self.verbose = obj.verbose
        self.idx_exc = obj.idx_exc # define 1 or 0 for the state which is subjected to exclusive constraints.
        self.idx_inc = obj.idx_inc # define 1 or 0 for the state which is subjected to box (inclusive) constraints.
        
        self.npx = 2*len(self.idx_exc) # number of axis aligned constraints
        self.safe_set = obj.safe_set # list of safe set in the format ('idx': [Al, bl, Au, bu]) for each polyhedron, it should be in order of idx_exc i.e 
        self.nb = obj.nb # number of boxes

        self.safe_set_constraint = obj.safe_set_constraint # apply safe set constraints

        self.input_rate_penalty = obj.input_rate_penalty                    # Introduce the integral action 
        self.bound_penalty   = obj.bound_penalty    # soft constraints to avoid unfeasibility
        
        self.terminal_goal = obj.terminal_goal # apply terminal goal constraints
        self.big_M = obj.big_M
        self.eps_thresh = 1e-4
        self.inf_value = 1e16
        # Form the array for the bounds
        # self.xmin_inc       = obj.xmin_inc      # min limit on states
        # self.xmax_inc       = obj.xmax_inc      # max limit on states
        # self.xmin_exc       = obj.xmin_exc      # min limit on states
        # self.xmax_exc       = obj.xmax_exc      # max limit on states
        min_scalar = 1.0
        max_scalar = 1.0
        self.xmin_inc       = [x * min_scalar for x in obj.xmin_inc]      # min limit on states
        self.xmax_inc       = [x * max_scalar for x in obj.xmax_inc]      # max limit on states
        self.xmin_exc       = [x * min_scalar for x in obj.xmin_exc]      # min limit on states
        self.xmax_exc       = [x * max_scalar for x in obj.xmax_exc]      # max limit on states


 
        # self.umin       = obj.umin      # min limit on control input  
        # self.umax       = obj.umax      # max limit on control input
        # self.dumin      = obj.dumin     # min limit on control input rate 
        # self.dumax      = obj.dumax     # max limit on control input rate
        

        self.umin       = [x * min_scalar for x in obj.umin]      # min limit on control input  
        self.umax       = [x * max_scalar for x in obj.umax]      # max limit on control input
        self.dumin      = [x * min_scalar for x in obj.dumin]     # min limit on control input rate 
        self.dumax      = [x * max_scalar for x in obj.dumax]     # max limit on control input rate
        
        self.Qx  = obj.Qx               # list: penalty on states 
        self.Qu  = obj.Qu               # list: Penalty on input (accelcycle, steer)
        if obj.input_rate_penalty:
            self.Qdu = obj.Qdu          # list: Penalty on Input rate 
        if obj.bound_penalty:
            self.Qew = obj.Qew          # list: Penalty on getting closer to bound 
     
        # MPC setting: 
        self.N     = obj.N              # N: Horizon, +1 because of state x0
        self.nx    = self.Qx.shape[0]   # nx: no. of states
        self.nu    = self.Qu.shape[0]   # nu: no.of control input
        
        # Define the system dynamics
        self.dt = obj.dt
        self.A = obj.A
        self.B = obj.B

        self.warm_start = obj.warm_start
        self.xPred = np.array([[0.0]*self.nx]*(self.N+1))     # states predicted from MPC self.solution                
        self.uPred = np.array([[0.0]*self.nu]*self.N)         # input predicted from MPC self.solution    
        self.uminus1 = np.array([0]*self.nu).T                # Past input which will be used when integral action is needed
        self.feasible = 0                                     # self.solution feasible == 1
        self.x0_local = np.array([0.]*self.nx)

    def setup(self, x0, xg = [], xref = [], uref = []):

        '''
        MPC formulation setup >>>
        Objective function:: Ju = JQx + JQu + JQdu + JQe  
        
        where,
        JQx  : Cost of states 
        JQu  : Cost of input
        JQdu : Cost of input rate
        JQe  : Cost of soft constraints
        
        Cost function changed to QP problem:
        min 1/2(z^T.P.z) + q^Tz
        subject to:
            l =< Az <= u
        
        Development Reference::
            Chapter 8: https://upcommons.upc.edu/bitstream/handle/2117/349185/tfm-shivam.pdf

        The sparse matrix for the nonlinear dynamics
        (A_vec, B_vec) is set to '1' at all places where the values are going
        to be changed. In the further update step this value will be updated.
        During the update the A matrix will be replaced with the new one by
        passing only the required value at those location. Follow this link
        for further information on how to update sparse matrix using OSQP
        [1]: https://groups.google.com/g/osqp/c/ZFvblAQdUxQ,
        [2]: https://math.stackexchange.com/questions/2256241/writing-a-convex-quadratic-program-qp-as-a-semidefinite-program-sdp
        '''
        self.objective(xref, uref)
        self.constraints(x0, xg)
        # self.q = self.q
        [N,nx,nu, nb] = self.N, self.nx, self.nu, self.nb 
        idx_inc = self.idx_inc
        idx_exc = self.idx_exc
        N_inc = len(self.idx_inc)
        N_exc = len(self.idx_exc)
        Npx = self.npx*N #cardinality of p vector

        #################################### Problem Setup ###################################
        if self.opt_method_type == 'osqp':
            self.prob = osqp.OSQP()         #osqp problem initialization
            self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=self.warm_start, polish=True, verbose = False, linsys_solver = "qdldl")
        
        if self.opt_method_type == 'milp_osqp':
            print ("Solver: MIQP_OSQP")
            self.prob = miosqp.MIOSQP()         #osqp problem initialization
            miosqp_settings = {
                    # integer feasibility tolerance
                    'eps_int_feas': 1e-09,
                    # maximum number of iterations
                    'max_iter_bb': 10000,
                    # tree exploration rule
                    #   [0] depth first
                    #   [1] two-phase: depth first until first incumbent and then  best bound
                    'tree_explor_rule': 0,
                    # branching rule
                    #   [0] max fractional part
                    'branching_rule': 0,
                    'verbose': False,
                    'print_interval': 1}

            osqp_settings = {'eps_abs': 1e-03,
                'eps_rel': 1e-09,
                'eps_prim_inf': 1e-03,
                'verbose': False}
            # print ("self.P.shape", self.P.shape, "self.q.shape", self.q.shape, "self.A.shape", self.A.shape, "self.l.shape", self.l.shape, "self.u.shape", self.u.shape)
            # print ("N*nx", N*nx, "(N-1)*nu", (N-1)*nu, "2*N*N_exc", 2*N*N_exc, "N*self.npx", N*self.npx)
            self.i_idx = np.arange(self.A.shape[1] - self.nb*N*self.npx, self.A.shape[1]) 

            # print ("len(self.i_idx)", len(self.i_idx), "self.i_idx", self.i_idx)
            # print ("self.P" , self.P.toarray(), "self.q", self.q, "self.A", self.A.toarray(), "self.l", self.l, "self.u", self.u)
            self.i_l = np.zeros(len(self.i_idx), dtype = np.int8)
            self.i_u = np.ones(len(self.i_idx), dtype = np.int8)
            # print ("self.q", self.q.shape)
            # self.q = self.q.squeeze()
            self.prob.setup(self.P, self.q, self.A, self.l, self.u, self.i_idx, self.i_l, self.i_u, miosqp_settings, osqp_settings)
        if self.opt_method_type == 'gurobi':
            # print ("self.P" , self.P.toarray(), "self.q", self.q, "self.A", self.A.toarray(), "self.l", self.l, "self.u", self.u)

            self.i_idx = np.arange(self.A.shape[1] - self.nb*N*self.npx, self.A.shape[1]) 
            self.i_l = np.zeros(len(self.i_idx))
            self.i_u = np.ones(len(self.i_idx))
            # self.prob = mpbpy.QuadprogProblem(self.P, self.q, self.A, self.l, self.u, self.i_idx, self.i_l, self.i_u)

        ### raw gurobi setup
        if self.opt_method_type == 'raw_gurobi':
            m = self.nb*self.nx*self.N
            Nx = self.N*self.nx
            Nu = (self.N-1)*self.nu
            Nxu = Nx + Nu
            Nt = Nx + Nu + m
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', self.verbose)
                env.setParam('DualReductions', 0)
                env.start()
                self.rg_model = gp.Model("qp", env=env)
                self.rg_x = self.rg_model.addMVar(Nxu + m, vtype=[GRB.CONTINUOUS] * Nxu + [GRB.BINARY] * m, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
                self.rg_objective = self.rg_x.T@self.P@self.rg_x + self.q.T@self.rg_x
                self.rg_model.setObjective(self.rg_objective, GRB.MINIMIZE)
                # self.rg_model.setParam('MIPGap', 1e-11)
                self.rg_constr1 = self.rg_model.addConstr( self.A@self.rg_x >= self.replace_inf(self.l), "c1")
                self.rg_constr2 = self.rg_model.addConstr( self.A@self.rg_x <= self.replace_inf(self.u), "c2")
                
                self.solve_raw_gurobi()

    def replace_inf(self, arr, large_positive=1e23, large_negative=-1e7):

        # Replace np.inf with large_positive and -np.inf with large_negative
        arr = np.where(arr == np.inf, GRB.INFINITY, arr)
        arr = np.where(arr == -np.inf, -GRB.INFINITY, arr)
        # arr
        return arr
    #####################################################################################################

    def objective(self, xr, ur):
        '''
        MPC formulation setup >>>
        Objective function:: Ju = JQx + JQu + JQdu + JQe  
        
        where,
        JQx  : Cost of states 
        JQu  : Cost of input
        JQdu : Cost of input rate
        JQe  : Cost of soft constraints
        
        Cost function changed to QP problem:
        min 1/2(z^T.P.z) + q^Tz
        subject to:
            l =< Az <= u
        '''

        [N,nx,nu, npx, nbx] = self.N, self.nx, self.nu, self.npx, self.nb 

        #################### P formulation ################################
        self.Qx  = sparse.diags(self.Qx)
        self.QxN = self.Qx
        self.Qu  = sparse.diags(self.Qu)
        
        PQx = sparse.block_diag([sparse.kron(sparse.eye(N-1), self.Qx), self.QxN], format='csc')
        PQu = sparse.kron(sparse.eye(N-1), self.Qu)
            
        #################### q formulation ################################
        if len(xr) > 0:
            self.qQx  = np.hstack([np.concatenate([-self.Qx.dot(xr_) for xr_ in xr]), -self.QxN.dot(xr[-1])])
        else:
            xr = np.array([0.0]*self.nx)
            self.qQx  = np.hstack([np.kron(np.ones(N-1), -self.Qx.dot(xr)), -self.QxN.dot(xr)])
            
        if len(ur) > 0:
            qQu  = np.hstack(np.concatenate([-self.Qu.dot(ur_) for ur_ in ur]))
        else:
            ur = np.array([0.0]*self.nu)
            qQu  = np.kron(np.ones(N-1), -self.Qu.dot(ur))
                                                    
        '''Objective function formulation'''
        if self.bound_penalty and self.input_rate_penalty:
            #################### P formulation ################################
            self.Qdu = sparse.diags(self.Qdu)
            idu = (2 * sparse.eye(N-1) - sparse.eye(N-1, k=1) - sparse.eye(N-1, k=-1))
            PQdu = sparse.kron(idu, self.Qdu)
            self.Qeps  = sparse.diags(self.Qew)
            PQeps = sparse.kron(sparse.eye(N), self.Qeps)
            self.P = sparse.block_diag([PQx, PQu + PQdu, PQeps], format='csc')
        
            #################### q formulation ################################
            qQdu = np.hstack([-self.Qdu.dot(self.uminus1), np.zeros(((N - 2) * nu, (N - 2) * nu))])
            self.Qeps  = np.diags(self.Qew)
            qQeps = np.zeros((N*nx, N*nx))
            self.q = np.hstack([self.qQx, qQu + qQdu, qQeps])

        elif self.input_rate_penalty:
            #################### P formulation ################################
            self.Qdu = sparse.diags(self.Qdu)
            idu = (2 * sparse.eye(N-1) - sparse.eye(N-1, k=1) - sparse.eye(N-1, k=-1))
            PQdu = sparse.kron(idu, self.Qdu)
            self.P = sparse.block_diag([PQx, PQu + PQdu], format='csc')
            
            #################### q formulation ################################
            # self.Qdu.dot(self.uminus1)
            qQdu = np.hstack([-self.Qdu.dot(self.uminus1), np.zeros(((N - 2) * nu, (N - 2) * nu))])
            self.q = np.hstack([self.qQx, qQu + qQdu])

        elif self.bound_penalty:
            #################### P formulation ################################
            self.Qeps  = sparse.diags(self.Qew)
            PQeps = sparse.kron(sparse.eye(N), self.Qeps)
            self.P = sparse.block_diag([PQx, PQu, PQeps], format='csc')

            #################### q formulation ################################
            qQeps = np.zeros((N*nx, N*nx))
            self.q = np.hstack([self.qQx, qQu, qQeps])

        else:
            self.P = sparse.block_diag([PQx, PQu], format='csc')
            self.q = np.hstack([self.qQx, qQu])

        # extend column of matrix P and vector q for box constraint variable "p"
        if self.npx:
            self.P = sparse.block_diag([self.P, np.zeros((nbx*npx*N,nbx*npx*N))]).tocsc()
            # print ("self.q,", self.q.shape, "np.zeros((1, npx*N))", np.zeros((1, nbx*npx*N)).shape)
            self.q = np.hstack([self.q, np.zeros(nbx*npx*N)])
        # self.P = self.P
    def constraints(self, x0, xg):
        '''
        Merge the MPC equality and inequality constraints, i.e. Combining equivalent and inequivalent constraints:
        l =< Az <= u
        '''
        self.equality_constraints(x0, xg)
        self.inequality_constraints()

        self.A = sparse.vstack([self.Aeq, self.Aineq]).tocsc()
        self.l = np.hstack([self.leq, self.lineq])
        self.u = np.hstack([self.ueq, self.uineq])

        ''' vector of non zero element is needed to update the sparse matrix during the Pq updates and bounds '''
        At = self.A.transpose(copy=True) 
        At.sort_indices()
        (self.A_col_indices, self.A_row_indices) = At.nonzero()

    def equality_constraints(self, x0, xg = []):
        '''Equality constraints : leq = Aeqz = ueq'''

        [N,nx,nu] = self.N, self.nx, self.nu 
        n_eps = N * nx
        Npx = self.npx*N #cardinality of p vector
        nbx = self.nb # number of boxes

        Ad = sparse.kron(sparse.eye(N), -sparse.eye(nx)) + sparse.kron(sparse.eye(N, k=-1), self.A)
        iBu = sparse.vstack([sparse.csc_matrix((1, N-1)), sparse.eye(N-1)])
        Bd = sparse.kron(iBu, self.B)
        
        self.Aeq = sparse.hstack([Ad, Bd]).tocsc()
        self.leq = np.hstack([-x0, np.zeros((N-1)*nx)])
        self.ueq = self.leq
        
        # if there is terminal constraint
        if xg.size:
            self.Axg = sparse.hstack([sparse.csr_matrix((nx,nx*(N-1))), sparse.eye(nx,nx), sparse.csr_matrix((nx,nu*(N-1)))]).tocsc()
            # self.Aeq = sparse.lil_matrix(sparse.hstack([Ax, Bu]))
            self.Aeq = sparse.vstack([self.Aeq, self.Axg]).tocsc().toarray()
            self.leq = np.hstack([self.leq, xg])
            self.ueq = self.leq
        
        # if bound penalty is applied
        if self.bound_penalty:
            self.Aeq = sparse.hstack([self.Aeq, sparse.csr_matrix((self.Aeq.shape[0], n_eps))]).tocsc()

        # for decision variable p
        self.Aeq = sparse.hstack([self.Aeq, sparse.csr_matrix((self.Aeq.shape[0], nbx*Npx))]).tocsc()
        
        # constraint for keeping all control input equivalent over horizon
        # iden = sparse.eye((N-2)*nu)
        # iden_left = sparse.hstack([iden, sparse.csc_matrix((iden.shape[0], nu))])
        # iden_right = -sparse.hstack([sparse.csc_matrix((iden.shape[0], nu)), iden])
        # self.Gueq_iden = iden_left + iden_right
        # # print ("Gu.shape", Gu.toarray())
        # Gueq = sparse.hstack([sparse.csr_matrix((self.Gueq_iden.shape[0], nx*N)), self.Gueq_iden, sparse.csr_matrix((self.Gueq_iden.shape[0], nbx*Npx))])
        # self.Gueq = Gueq
        # # print ("Gueq.shape", Gueq.shape, "self.Aeq.shape", self.Aeq.shape)
        # self.Aeq = sparse.vstack([self.Aeq, Gueq]).tocsc()
        # self.leq = np.hstack([self.leq, np.zeros(Gueq.shape[0])])
        # self.ueq = np.hstack([self.ueq, np.zeros(Gueq.shape[0])])

    def inequality_constraints(self):
        '''Inequality constraints: bound on states
        l2 < A2z < u2'''
        
        [N,nx,nu] = self.N, self.nx, self.nu 
        n_eps = N * nx
        idx_inc = self.idx_inc
        idx_exc = self.idx_exc
        N_inc = len(self.idx_inc)
        N_exc = len(self.idx_exc)
        Npx = self.npx*N #cardinality of p vector
        nbx = self.nb # number of boxes

        '''Inequality constraints: inclusive constraints'''
        ''' xl <= Gx + eps_x <= xu ''' 
        g = np.zeros((N_inc, nx))
        g[range(N_inc), idx_inc] = 1
        G_x_inc = sparse.hstack([sparse.kron(sparse.eye(N), g), sparse.csc_matrix(((N)*g.shape[0], (N-1)*nu))]).tocsc()
        
        # if epsilon is added for bound penalty
        if self.bound_penalty:
            G_x_inc = sparse.hstack([G_x_inc, sparse.eye(n_eps)]).tocsc()
        
        ## added for p vector
        self.G_x_inc = sparse.hstack([G_x_inc, sparse.csr_matrix((G_x_inc.shape[0], nbx*Npx))]).tocsc()
        self.l_x_inc = np.kron(np.ones(N), self.xmin_inc) # lower bound of inequalities on states
        self.u_x_inc = np.kron(np.ones(N), self.xmax_inc) # upper bound of inequalities on states

        '''Inequality constraints: exclusive constraints [BOX CONSTRAINTS NON-CONVEX TYPE]'''
        '''gx + eps_x <= xl,  xu <= gx + eps_x
        Converted to -inf <= Gx_vec - Mpvec + eps_xvec <= xbvec '''  
        if len(self.xmin_exc) > 0:
            # print ("gx + eps_x <= xl,  xu <= gx + eps_x")
            print ("gx + eps_x <= xl,  xu <= gx + eps_x")
            g = np.zeros((N_exc, nx))
            g[range(N_exc), idx_exc] = 1
            G_x_exc = sparse.kron(sparse.eye(N), g)
            G_x_exc = sparse.hstack([G_x_exc, sparse.csc_matrix((G_x_exc.shape[0], (N-1)*nu))]).tocsc()
            G_x_exc = sparse.vstack([G_x_exc, -G_x_exc]).tocsc()
            
            # if epsilon is added for bound penalty
            if self.bound_penalty:
                G_x_exc = sparse.hstack([G_x_exc, sparse.eye(n_eps)]).tocsc()
            G_x_exc_ = []
            G_p_exc_ = []
            lineq_x_exc_ = []
            uineq_x_exc_ = []
                
            for xmin_exc_, xmax_exc_ in zip(self.xmin_exc, self.xmax_exc):
                ## added for p vector
                G_p_excl = sparse.hstack([-self.big_M*sparse.eye((int(Npx/2))), sparse.csc_matrix((int(Npx/2), int(Npx/2)))]).tocsc()
                G_p_excu = sparse.hstack([sparse.csc_matrix((int(Npx/2), int(Npx/2))), -self.big_M*sparse.eye((int(Npx/2)))]).tocsc()
                G_p_exc = sparse.vstack([G_p_excl, G_p_excu]).tocsc()
                lineq_x_exc = np.array(np.kron(np.ones(N), xmin_exc_)) # lower bound for box
                uineq_x_exc = np.array(np.kron(np.ones(N), xmax_exc_)) # upper bound for box
                uineq_x_exc = np.hstack([lineq_x_exc, -uineq_x_exc])
                lineq_x_exc = -np.inf*np.ones(uineq_x_exc.shape[0])
                G_p_exc_.append(G_p_exc)
                G_x_exc_.append(G_x_exc)
                lineq_x_exc_.append(lineq_x_exc)
                uineq_x_exc_.append(uineq_x_exc)

            self.G_p_exc = sparse.block_diag(G_p_exc_).tocsc()
            G_x_exc = sparse.vstack(G_x_exc_).tocsc()
            self.G_x_exc = sparse.hstack([G_x_exc, self.G_p_exc]).tocsc()
            self.lineq_x_exc = np.hstack(lineq_x_exc_)
            self.uineq_x_exc = np.hstack(uineq_x_exc_)

            # print ("self.G_x_exc \n", self.G_x_exc.toarray(), "\n self.lineq_x_exc \n", self.lineq_x_exc, "\n self.uineq_x_exc \n", self.uineq_x_exc)


        '''Inequality constraints: sfae set inclusive and safe set exclusive constraints [OUTER BOX CONVEX TYPE, INNER BOX CONSTRAINTS NON-CONVEX TYPE]'''
        '''-inf <= Sx_vec - Mpvec <= bvec :: safe_set:-> {'idx': ['Al': Al, 'bl': bl, 'Au':Au, 'bu':bu}'''  
        res = 0 #-1e-1
        if self.safe_set_constraint:
            if not self.safe_set:
                print ("error provide safe set constraints")
                sys.exit()
            
            last_idx = len(self.safe_set) - 1

            # Safe set for outer box convex type
            safe_set_ = self.safe_set[last_idx]
            # Amat = safe_set_['H_struct'][0]
            # bvec = safe_set_['f_struct'][0].squeeze()
            Amat = np.array([
    [  99.502487562189,    0.0,   5.02487562189055,   0.0],
    [ -99.502487562189,    0.0,  -5.02487562189055,   0.0],
    [-140.716025919216,    0.0,   0.0,                0.0],
    [   0.0,            -140.716025919216, 0.0,       0.0],
    [   0.0,               0.0,  -7.07106781186547,   0.0],
    [   0.0,               0.0,   0.0,               -7.07106781186547],
    [ 140.716025919216,    0.0,   0.0,                0.0],
    [   0.0,             140.716025919216, 0.0,       0.0],
    [   0.0,               0.0,   7.07106781186547,   0.0],
    [   0.0,               0.0,   0.0,                7.07106781186547],
    [   0.0,             99.502487562189,  0.0,       5.02487562189055],
    [   0.0,            -99.502487562189,  0.0,      -5.02487562189055],
], dtype=np.float64)
            bvec = np.array([
    69.638050681592,
   -59.1515622189055,
   -83.567553997742,
    -0.0142231537518366,
     0.217713935286649,
     0.302566183343612,
    98.6831681864456,
    24.0858540310226,
     0.246700363567832,
     0.325832188475671,
    17.1914691855721,
     0.123153050746269,
], dtype=np.float64)


            Sx_inc = sparse.hstack([sparse.kron(sparse.eye(N), Amat), np.zeros(((N)*Amat.shape[0], (N-1)*nu))]).tocsc()            
            # if epsilon is added for bound penalty
            if self.bound_penalty:
                Sx_inc = sparse.hstack([Sx_inc, sparse.eye(n_eps)]).tocsc()
            
            ## added for p vector
            self.Sx_inc = sparse.hstack([Sx_inc, sparse.csr_matrix((Sx_inc.shape[0], nbx*Npx))]).tocsc()
            self.usx_inc = np.kron(np.ones(N), bvec*(1 + res)) # upper bound of inequalities on states
            self.lsx_inc = -np.inf*np.ones(self.usx_inc.shape) # lower bound of inequalities on states
            print ("if len(self.xmin_exc) > 0:", len(self.xmin_exc) > 0)
            if len(self.xmin_exc) > 0:
                Sx_exc_ = []
                Sz_exc_ = []
                Sm_exc_ = []
                usineq_x_exc_ = []
                lsineq_x_exc_ = []
                # Safe set for inner box Non-convex type
                for i in range(len(self.safe_set)):
                    if last_idx != i:
                        safe_set_ = self.safe_set[i]
                        self.extract_safe_set(safe_set_)
                        # Sl = np.kron(np.eye(N), self.sAlmat).tocsc()
                        # Sl = np.hstack([Sl, sparse.csc_matrix((Sl.shape[0], (N-1)*nu))]).tocsc()
                        # Su = np.kron(np.eye(N), self.sAumat).tocsc()
                        # Su = np.hstack([Su, sparse.csc_matrix((Su.shape[0], (N-1)*nu))]).tocsc()
                        Sx_exc_.append(self.sAmat)
                        Sz_exc = sparse.csc_matrix((self.sAmat.shape[0], (N-1)*nu))

                        # if epsilon is added for bound penalty
                        if self.bound_penalty:
                            Sz_exc = sparse.hstack([Sz_exc, sparse.csc_matrix((Sx_exc.shape[0], N*nx))]).tocsc()
                        Sz_exc_.append(Sz_exc)

                        ## added for p vector
                        Sm_exc_.append(self.spmat)

                        usineq_x_exc_.append(self.sbvec*(1 + res))
                        lsineq_x_exc_.append(-np.inf*np.ones(self.sbvec.shape[0]))

                Sx_exc = sparse.vstack(Sx_exc_).tocsc()
                Sm_exc = sparse.block_diag(Sm_exc_).tocsc()
                Sz_exc = sparse.vstack(Sz_exc_).tocsc()
                self.Sx_exc = sparse.hstack([Sx_exc, Sz_exc, Sm_exc]).tocsc()
                self.usineq_x_exc = np.hstack(usineq_x_exc_)
                self.lsineq_x_exc = np.hstack(lsineq_x_exc_)


        '''Inequality constraints: bound on input'''
        Gineq_u = sparse.hstack([sparse.csc_matrix(((N-1)*nu, N*nx)), sparse.eye((N -1 )* nu)]).tocsc()
        if self.bound_penalty:
            Gineq_u = sparse.hstack([Gineq_u, sparse.csc_matrix((self.Gineq_u.shape[0], n_eps))]).tocsc()# For soft constraints slack variables

        ## added for p vector
        self.Gineq_u = sparse.hstack([Gineq_u, sparse.csc_matrix((Gineq_u.shape[0], nbx*Npx))]).tocsc()    
        self.lineq_u = np.kron(np.ones(N-1), self.umin)     # lower bound of inequalities on input
        self.uineq_u = np.kron(np.ones(N-1), self.umax)     # upper bound of inequalities on input

        '''Inequality constraints: bounds on du (input rate)'''
        if self.input_rate_penalty == True:
            Gineq_du = sparse.kron(sparse.eye((N-1)) - sparse.eye((N-1), k=-1), sparse.eye(nu))
            Gineq_du = sparse.hstack([sparse.csc_matrix((Gineq_du.shape[0], N*nx)), Gineq_du]).tocsc()
            if self.bound_penalty:  
                Gineq_du = sparse.hstack([Gineq_du, sparse.csc_matrix((Gineq_du.shape[0], n_eps))]).tocsc()
            
            ## added for p vector
            self.Gineq_du = sparse.hstack([Gineq_du, sparse.csc_matrix((Gineq_du.shape[0], nbx*Npx))]).tocsc()
            self.uineq_du = np.kron(np.ones(N-1), self.dumax)   # upper bound of inequalities on input rate
            self.lineq_du = np.kron(np.ones(N-1), self.dumin)   # lower bound of inequalities on input rate

            # self.uineq_du = np.kron(np.ones(N+1), self.dumax)   # upper bound of inequalities on input rate 
            self.uineq_du[0:nu] += self.uminus1[0:nu]           # Equality constraint on previous input uminus_1
            # self.lineq_du = np.kron(np.ones(N+1), self.dumin)   # lower bound of inequalities on input rate
            self.lineq_du[0:nu] += self.uminus1[0:nu]           # Equality constraint on previous input uminus_1
        
        '''Inequality constraints for binary variables for handeling exclusive constraints'''
        if len(self.xmin_exc) > 0:
            # print ("Inequality constraints for binary variables for handeling exclusive constraints")
            Gineq_p_ = []
            Gineq_p_x_ = []
            lineq_p_ = []
            uineq_p_ = []
            for xmin_exc_ in self.xmin_exc:
                g = np.ones(N_exc)
                Gineq_pl = sparse.kron(sparse.eye(N),g)
                # print ("Gineq_pl", Gineq_pl.shape)
                Gineq_p = sparse.hstack([Gineq_pl,Gineq_pl]).tocsc()
                if self.bound_penalty:
                    Gineq_p_x_.append(sparse.csc_matrix((Gineq_p.shape[0], N*nx + (N-1)*nu + n_eps)))
                else:
                    Gineq_p_x_.append(sparse.csc_matrix((Gineq_p.shape[0], N*nx + (N-1)*nu)))
                lineq_p = 2*np.ones(Gineq_p.shape[0])
                uineq_p = 3*np.ones(Gineq_p.shape[0])
                Gineq_p_.append(Gineq_p)
                lineq_p_.append(lineq_p)
                uineq_p_.append(uineq_p)
            
            Gineq_p_x = sparse.vstack(Gineq_p_x_).tocsc()
            Gineq_p = sparse.block_diag(Gineq_p_).tocsc()
            self.Gineq_p = sparse.hstack([Gineq_p_x, Gineq_p]).tocsc()
            
            self.lineq_p = np.hstack(lineq_p_)
            self.uineq_p = np.hstack(uineq_p_)
            # print ("self.Gineq_p \n", self.Gineq_p.toarray(), "\n self.lineq_p \n", self.lineq_p, "\n self.uineq_p \n", self.uineq_p)
            
        '''Inequality constraints: threshold on eps bound penalty'''
        self.Gineq_eps = sparse.hstack([sparse.csc_matrix((N*nx, N*nx + (N-1)*nu)), sparse.eye(n_eps), sparse.csc_matrix((N*nx,nbx*Npx))]).tocsc()
        self.lineq_eps = np.zeros(N*nx)
        self.uineq_eps = self.eps_thresh*np.ones(N*nx)    

        '''Combining all the inequality constraints'''

        if self.safe_set_constraint and self.bound_penalty and self.input_rate_penalty:
            self.Aineq = sparse.vstack([self.G_x_inc, self.G_x_exc, self.Sx_exc, self.Gineq_u, self.Gineq_du, self.Gineq_p, self.Gineq_eps]).tocsc()
            self.lineq = np.hstack([self.l_x_inc, self.lineq_x_exc, self.lsineq_x_exc, self.lineq_u, self.lineq_du, self.lineq_p, self.lineq_eps])
            self.uineq = np.hstack([self.u_x_inc, self.uineq_x_exc, self.usineq_x_exc, self.uineq_u, self.uineq_du, self.uineq_p, self.uineq_eps])
        
        elif self.bound_penalty:
            self.Aineq = sparse.vstack([self.G_x_inc, self.G_x_exc, self.Sx_exc, self.Gineq_u, self.Gineq_p, self.Gineq_eps]).tocsc()
            self.lineq = np.hstack([self.l_x_inc, self.lineq_x_exc, self.lsineq_x_exc, self.lineq_u, self.lineq_p, self.lineq_eps])
            self.uineq = np.hstack([self.u_x_inc, self.uineq_x_exc, self.usineq_x_exc, self.uineq_u, self.uineq_p, self.uineq_eps])

        elif self.input_rate_penalty:
            self.Aineq = sparse.vstack([self.G_x_inc, self.G_x_exc, self.Sx_exc, self.Gineq_u, self.Gineq_du, self.Gineq_p]).tocsc()
            self.lineq = np.hstack([self.l_x_inc, self.lineq_x_exc, self.lsineq_x_exc, self.lineq_u, self.lineq_du, self.lineq_p])
            self.uineq = np.hstack([self.u_x_inc, self.uineq_x_exc, self.usineq_x_exc, self.uineq_u, self.uineq_du, self.uineq_p]) 
        
        if self.safe_set_constraint:
            # print ("self.safe_set_constraint", self.safe_set_constraint)
            if len(self.xmin_exc) > 0:
                print ("considering safe set constraints")
                # self.Aineq = sparse.vstack([self.G_x_inc, self.Sx_inc, self.G_x_exc, self.Sx_exc, self.Gineq_u, self.Gineq_p]).tocsc()
                # self.lineq = np.hstack([self.l_x_inc, self.lsx_inc, self.lineq_x_exc, self.lsineq_x_exc, self.lineq_u, self.lineq_p])
                # self.uineq = np.hstack([self.u_x_inc, self.usx_inc, self.uineq_x_exc, self.usineq_x_exc, self.uineq_u, self.uineq_p])

                self.Aineq = sparse.vstack([self.Sx_inc, self.Sx_exc, self.Gineq_u, self.Gineq_p]).tocsc()
                self.lineq = np.hstack([self.lsx_inc, self.lsineq_x_exc, self.lineq_u, self.lineq_p])
                self.uineq = np.hstack([self.usx_inc, self.usineq_x_exc, self.uineq_u, self.uineq_p])

                # self.Aineq = sparse.vstack([self.G_x_inc, self.Sx_inc, self.G_x_exc, self.Gineq_u, self.Gineq_p]).tocsc()
                # self.lineq = np.hstack([self.l_x_inc, self.lsx_inc, self.lineq_x_exc, self.lineq_u, self.lineq_p])
                # self.uineq = np.hstack([self.u_x_inc, self.usx_inc, self.uineq_x_exc, self.uineq_u, self.uineq_p])

                # self.Aineq = sparse.vstack([self.G_x_inc, self.Sx_inc, self.G_x_exc, self.Gineq_u, self.Gineq_p]).tocsc()
                # self.lineq = np.hstack([self.l_x_inc, self.lsx_inc, self.lineq_x_exc, self.lineq_u, self.lineq_p])
                # self.uineq = np.hstack([self.u_x_inc, self.usx_inc, self.uineq_x_exc, self.uineq_u, self.uineq_p])

            else:  
                print ("this case") 
                # self.Aineq = sparse.vstack([self.G_x_inc, self.Sx_inc, self.Gineq_u]).tocsc()
                # self.lineq = np.hstack([self.l_x_inc, self.lsx_inc, self.lineq_u])
                # self.uineq = np.hstack([self.u_x_inc, self.usx_inc, self.uineq_u])

                self.Aineq = sparse.vstack([self.Sx_inc]).tocsc()
                self.lineq = np.hstack([self.lsx_inc])
                self.uineq = np.hstack([self.usx_inc])
                            
                # self.Aineq = sparse.vstack([self.G_x_inc, self.G_x_exc, self.Sx_exc, self.Gineq_u, self.Gineq_p]).tocsc()
                # self.lineq = np.hstack([self.l_x_inc, self.lineq_x_exc, self.lsineq_x_exc, self.lineq_u, self.lineq_p])
                # self.uineq = np.hstack([self.u_x_inc, self.uineq_x_exc, self.usineq_x_exc, self.uineq_u, self.uineq_p])

        elif self.idx_exc:
            # print ("G_x_exc", G_x_exc.toarray())
            print ("G_x_inc", self.G_x_inc.shape, "G_x_exc", self.G_x_exc.shape, "Gineq_u", self.Gineq_u.shape, "Gineq_p", self.Gineq_p.shape)
            self.Aineq = sparse.vstack([self.G_x_inc, self.G_x_exc, self.Gineq_u, self.Gineq_p]).tocsc()
            self.lineq = np.hstack([self.l_x_inc, self.lineq_x_exc, self.lineq_u, self.lineq_p])
            self.uineq = np.hstack([self.u_x_inc, self.uineq_x_exc, self.uineq_u, self.uineq_p])

        else:
            self.Aineq = sparse.vstack([self.G_x_inc, self.Gineq_u]).tocsc()
            self.lineq = np.hstack([self.l_x_inc, self.lineq_u])
            self.uineq = np.hstack([self.u_x_inc, self.uineq_u])

    def extract_safe_set(self, safe_set):
        # safe_set:-> {'idx': ['Al': Al, 'bl': bl, 'Au':Au, 'bu':bu}
        idx_exc = self.idx_exc
        N_exc = len(idx_exc)
         
        nx = self.nx
        N = self.N
        
        counter = 0
        for i in idx_exc:
            Al = safe_set[i]['Al'][0]#[:2, :]
            bl = safe_set[i]['bl'][0]#[:2, :]
            Au = safe_set[i]['Au'][0]#[:2, :]
            bu = safe_set[i]['bu'][0]#[:2, :]
            # print ("Al", Al.shape, "bl", bl.shape, "Au", Au.shape, "bu", bu.shape)
            
            if Al.shape[0] < Au.shape[0]:
                Al = np.vstack([Al, np.zeros((Au.shape[0] - Al.shape[0], Al.shape[1]) )])
                bl = np.vstack([bl, np.zeros((Au.shape[0] - bl.shape[0], bl.shape[1]) )])
            elif Al.shape[0] > Au.shape[0]:
                Au = np.vstack([Au, np.zeros((Al.shape[0] - Au.shape[0], Au.shape[1]) )])
                bu = np.vstack([bu, np.zeros((Al.shape[0] - bu.shape[0], bu.shape[1]) )])

            # for p vector 
            g = np.zeros(len(idx_exc))
            g[i] = 1
            pl = np.tile(g, (Al.shape[0],1))
            pu = np.tile(g, (Au.shape[0],1))
            
            if Al.shape[1] != nx:
                Al = sparse.csc_matrix((Al.shape[0], nx))*0
                # bl = sparse.csc_matrix((bl.shape[0], 1))*0
                bl = np.zeros((Al.shape[0], 1))
                bl = sparse.csc_matrix(bl)

            if Au.shape[1] != nx:
                Au = sparse.csc_matrix((Au.shape[0], nx))*0
                # bu = sparse.csc_matrix((bu.shape[0], 1))*0
                bu = np.zeros((Au.shape[0], 1))
                bu = sparse.csc_matrix(bu)
                
            # print ("Al", Al.shape, "bl", bl.shape, "Au", Au.shape, "bu", bu.shape, "pl", pl.shape, "pu", pu.shape)
            if counter == 0:
                Almat = sparse.csr_matrix(Al)
                blvec = sparse.csr_matrix(bl)
                Aumat = sparse.csr_matrix(Au)
                buvec = sparse.csr_matrix(bu)
                plmat = sparse.csr_matrix(-abs(np.max(bl))*pl)
                pumat = sparse.csr_matrix(-abs(np.max(bu))*pu)

                
            else:
                Almat = sparse.vstack([Almat, Al])
                blvec = sparse.vstack([blvec, bl])
                Aumat = sparse.vstack([Aumat, Au])
                buvec = sparse.vstack([buvec, bu])
                plmat = sparse.vstack([plmat, -abs(np.max(bl))*pl])
                pumat = sparse.vstack([pumat, -abs(np.max(bu))*pu])
            counter += 1
        eps = 1e-9
        self.sAlmat = sparse.kron(np.eye(N), Almat)
        self.sAumat = sparse.kron(np.eye(N), Aumat)
        self.splmat = sparse.kron(np.eye(N), plmat)
        self.spumat = sparse.kron(np.eye(N), pumat)
        self.sAmat = sparse.vstack([self.sAlmat, self.sAumat]).tocsc()
        self.spmat = sparse.block_diag([self.splmat, self.spumat]).tocsc()
        blvec = blvec.toarray().squeeze()
        self.blvec = np.kron(np.ones(N), blvec)
        buvec = buvec.toarray().squeeze()
        self.buvec = np.kron(np.ones(N), buvec)
        self.sbvec = np.concatenate([self.blvec, self.buvec]).squeeze() #*(1 + eps)
        # self.sbvec = sbvec.toarray().squeeze()

    def update(self, x0, xg = np.array([]), xref = [], uref = []):
        '''
        Input: LPV Model(A_vec, B_vec) , current estimated state states (x0), previous input if slew_rate == on 
        Ouput: Update the MPC formulation
        '''

        [N,nx,nu] = self.N, self.nx, self.nu 

        if len(xref) > 0:
            self.qQx  = np.hstack([np.concatenate([-self.Qx.dot(xr_) for xr_ in xref]), -self.QxN.dot(xr[-1])])
            self.q[:self.qQx.shape[0]] = self.qQx

        if len(uref) > 0:
            qQu  = np.hstack([np.concatenate([-self.Qu.dot(ur_) for ur_ in uref])])
            # print ("qQu.shape", (qQu).shape, "self.q.shape", self.q.shape, "self.q[self.qQx.shape[0]:self.qQx.shape[0] + qQu.shape[0]]", (self.q[self.qQx.shape[0]:self.qQx.shape[0] + qQu.shape[0]]).shape)
            self.qQu = qQu
            self.q[self.qQx.shape[0]:self.qQx.shape[0] + qQu.shape[0]] = qQu

            ############### Update vector q  ##############        
        
        self.l[:nx] = -x0
        self.u[:nx] = -x0


        if xg.size:
            self.l[nx*(N): nx*(N+1)] = xg
            self.u[nx*(N): nx*(N+1)] = xg

        ########### Update inequality constraint if slew_rate == on with previous input (uminus_1)##############
        
        
        if self.bound_penalty and self.input_rate_penalty and self.safe_set_constraint:
            ndu = self.u_x_inc.shape[0] + self.uineq_x_exc.shape[0] + self.usineq_x_exc.shape[0] + self.uineq_u.shape[0]
            self.lineq[ndu: ndu + nu] = self.dumin + self.uminus1[0:nu]
            self.uineq[ndu: ndu + nu] = self.dumax + self.uminus1[0:nu]

        #     self.Aineq = sparse.vstack([self.G_x_inc, self.G_x_exc, self.Sx_exc, self.Gineq_u, self.Gineq_du, self.Gineq_p, self.Gineq_eps]).tocsc()
        #     self.lineq = np.hstack([self.l_x_inc, lineq_x_exc, lsineq_x_exc, lineq_u, lineq_du, lineq_p, lineq_eps])
        #     self.uineq = np.hstack([self.u_x_inc, uineq_x_exc, usineq_x_exc, uineq_u, uineq_du, uineq_p, uineq_eps])
        
        elif self.bound_penalty:
            ndu = self.u_x_inc.shape[0] + self.uineq_x_exc.shape[0] + self.usineq_x_exc.shape[0] + self.uineq_u.shape[0]
            self.lineq[ndu: ndu + nu] = self.dumin + self.uminus1[0:nu]
            self.uineq[ndu: ndu + nu] = self.dumax + self.uminus1[0:nu]

        #     self.Aineq = sparse.vstack([self.G_x_inc, self.G_x_exc, self.Sx_exc, self.Gineq_u, self.Gineq_p, self.Gineq_eps]).tocsc()
        #     self.lineq = np.hstack([self.l_x_inc, self.lineq_x_exc, self.lsineq_x_exc, self.lineq_u, self.lineq_p, self.lineq_eps])
        #     self.uineq = np.hstack([self.u_x_inc, self.uineq_x_exc, self.usineq_x_exc, self.uineq_u, self.uineq_p, self.uineq_eps])
# 
        elif self.input_rate_penalty and self.safe_set_constraint:
            ndu = self.u_x_inc.shape[0] + self.uineq_x_exc.shape[0] + self.usineq_x_exc.shape[0] + self.uineq_u.shape[0]
            self.lineq[ndu: ndu + nu] = self.dumin + self.uminus1[0:nu]
            self.uineq[ndu: ndu + nu] = self.dumax + self.uminus1[0:nu]

        #     self.Aineq = sparse.vstack([self.G_x_inc, self.G_x_exc, self.Sx_exc, self.Gineq_u, self.Gineq_du, self.Gineq_p]).tocsc()
        #     self.lineq = np.hstack([self.l_x_inc, self.lineq_x_exc, self.lsineq_x_exc, self.lineq_u, self.lineq_du, self.lineq_p])
        #     self.uineq = np.hstack([self.u_x_inc, self.uineq_x_exc, self.usineq_x_exc, self.uineq_u, self.uineq_du, self.uineq_p]) 
        


        if self.opt_method_type == 'osqp':
            self.prob.update(q = self.q , l= self.l, u= self.u)

        elif self.opt_method_type == 'milp_osqp':
            self.prob.update_vectors(q = self.q , l= self.l, u= self.u)

        elif self.opt_method_type == 'gurobi':
            self.prob.q = self.q
            self.prob.l = self.l
            self.prob.u = self.u

        elif self.opt_method_type == 'raw_gurobi':
            self.rg_objective = self.rg_x.T@self.P@self.rg_x + self.q.T@self.rg_x
            self.rg_model.setObjective(self.rg_objective, GRB.MINIMIZE)
            self.rg_constr1.RHS = self.replace_inf(self.l)
            self.rg_constr2.RHS = self.replace_inf(self.u)
            self.rg_x.start = self.rg_prev_soln

    def solve(self):
        '''
        Solve the QP problem 
        '''

        if self.opt_method_type == 'osqp':
            self.solve_osqp()

        if self.opt_method_type == 'milp_osqp':
            self.solve_milp_osqp()

        elif self.opt_method_type == 'scipy':
            self.solve_scipy(x0)

        elif self.opt_method_type == 'gurobi':
            self.solve_gurobi()
        
        elif self.opt_method_type == 'raw_gurobi':
            feasible = self.solve_raw_gurobi()

        else:
            print ("Solver not implemented")
            sys.exit()
        
        return feasible
        ## CVXPY can not be implemented, it doesn't support symbolic nonlinearity 
        # elif self.opt_method_type == 'cvxpy':

    ##################### OSQP ADMM Solver #########################
    def solve_osqp(self):
        '''
        Solve the QP problem 
        '''
        [N,nx,nu] = self.N, self.nx, self.nu 

        self.res = self.prob.solve()        
        # Check solver status
        if self.res.info.status != 'solved':
            print ('OSQP did not solve the problem!')
            self.feasible = 0
            self.xPred = np.array([])
            self.uPred = np.array([])
        else:
            self.feasible = 1
            self.Solution = self.res.x
            self.rg_prev_soln = self.Solution
            self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(nx * (N))]), (N, nx)))).T
            self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[nx * (N) + np.arange(nu * (N-1))]), (N-1, nu)))).T
            # need to fix dimension for p vector
            self.pPred = np.squeeze(np.transpose(np.reshape((self.Solution[nx * (N) + nu * (N-1) + np.arange(self.N*self.npx)]), (N, self.npx)))).T

    ##################### MILP OSQP ADMM Solver #########################
    def solve_milp_osqp(self):
        '''
        Solve the QP problem 
        '''
        print ("solving using Milp OSQP")
        [N,nx,nu] = self.N, self.nx, self.nu 

        self.res = self.prob.solve()        
        # Check solver status
        if self.res.status != 'Solved':
            print ('MILP OSQP did not solve the problem!')
            self.feasible = 0
            self.xPred = np.array([])
            self.uPred = np.array([])
            self.pPred = np.array([])
        else:
            self.feasible = 1
            self.Solution = self.res.x

            self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(nx * (N))]), (N, nx)))).T
            self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[nx * (N) + np.arange(nu * (N-1))]), (N-1, nu)))).T
            # need to fix dimension for p vector
            self.pPred = np.squeeze(np.transpose(np.reshape((self.Solution[nx * (N) + nu * (N-1) + np.arange(self.N*self.npx*self.nb)]), (N, self.npx, self.nb)))).T

    def solve_gurobi(self):
        '''
        Solve the QP problem 
        '''
        [N,nx,nu] = self.N, self.nx, self.nu 

        # self.res = self.prob.solve(solver=mpbpy.GUROBI, verbose=False, Threads=1)
        # self.res = self.prob.solve(solver=mpbpy.CPLEX, verbose=False)
        # Check solver status
        if self.res.status != 'optimal':
            print ('Gurobi did not solve the problem!')
            self.feasible = 0
            self.xPred = np.array([])
            self.uPred = np.array([])
        else:
            self.feasible = 1
            self.Solution = self.res.x
            self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(nx * (N))]), (N, nx)))).T
            self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[nx * (N) + np.arange(nu * (N-1))]), (N-1, nu)))).T
            # need to fix dimension for p vector
            self.pPred = np.squeeze(np.transpose(np.reshape((self.Solution[nx * (N) + nu * (N-1) + np.arange(self.N*self.npx*self.nb)]), (N, self.npx, self.nb)))).T

    def solve_raw_gurobi(self):
        '''
        Solve the QP problem 
        '''
        [N,nx,nu, nb] = self.N, self.nx, self.nu, self.nb 

        self.rg_model.optimize()
        print ("self.rg_model.status", self.rg_model.status)
        # Check solver status
        if self.rg_model.status != GRB.OPTIMAL:
            print ('RAW Gurobi did not solve the problem!')
            # self.rg_model.computeIIS()
            # self.rg_model.write("IIS.ilp")
            # print("Model infeasible. Inspect IIS given in IIS.ilp file.")

            feasible = 0
            m = self.nb*self.nx*self.N
            Nx = self.N*self.nx
            Nu = (self.N-1)*self.nu
            Nxu = Nx + Nu
            Nt = Nxu + m
            self.rg_prev_soln = np.zeros(Nt)
            self.xPred = np.array([])
            self.uPred = np.array([])
        else:
            feasible = 1

            self.Solution = self.rg_x.X
            self.rg_prev_soln = self.Solution
            self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(nx * (N))]), (N, nx)))).T
            self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[nx * (N) + np.arange(nu * (N-1))]), (N-1, nu)))).T
            # need to fix dimension for p vector
            # print ("self.Solution[nx * (N) + nu * (N-1) + np.arange(self.N*self.npx)]", self.Solution[nx * (N) + nu * (N-1) + np.arange(self.N*self.npx*self.nb)])

            self.pPred = self.Solution[nx * (N) + nu * (N-1) + np.arange(self.N*self.npx*self.nb)]
            pPred = self.pPred.reshape(self.nb, self.N*self.npx)
            self.pPredl = pPred[:, :self.N*self.npx//2]
            self.pPredu = pPred[:, self.N*self.npx//2:]
        return feasible
    
    ##################### scipy #########################
    def solve_scipy(self, x0):
        '''
        Solve the QP problem 
        '''
        [N,nx,nu] = self.N, self.nx, self.nu 


        eq_cons = {'type': 'eq',
           'fun' : self.model_constraint}

        bnds = Bounds(np.hstack([self.lineq_x, self.lineq_u]), np.hstack([self.uineq_x, self.uineq_u]))
        initialization = np.concatenate([x0,self.xPred[1:], self.uPred], axis=None)
        self.res = opt.minimize(self.objective_function, initialization, bounds = bnds, method = 'SLSQP', constraints = eq_cons)

        if self.res.success == False:
            print ('OSQP did not solve the problem!')
            self.feasible = 0
        self.Solution = self.res.x

        self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(nx * (N + 1))]), (N + 1, nx)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[nx * (N + 1) + np.arange(nu * N)]), (N, nu)))).T

    def model_constraint(self, vec):
        return self.Aeq@vec - self.leq

    def objective_function(self, vec):
        return 1/2*vec.T@self.P@vec + vec.T@self.q

    ############# CVXPY #################################
    def cvxpy_setup(self):
        # Define problem
        u = Variable((nu, N))
        x = Variable((nx, N+1))
        x_init = Parameter(nx)
        objective = 0
        constraints = [x[:,0] == x_init]
        for k in range(N):
            objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k], R)
            constraints += [x[:,k+1] == Ad*x[:,k] + Bd*u[:,k]]
            constraints += [xmin <= x[:,k], x[:,k] <= xmax]
            constraints += [umin <= u[:,k], u[:,k] <= umax]
        objective += quad_form(x[:,N] - xr, QN)
        prob = Problem(Minimize(objective), constraints)

        # Simulate in closed loop
        nsim = 15
        for i in range(nsim):
            x_init.value = x0
            prob.solve(solver=OSQP, warm_start=False)
            x0 = Ad.dot(x0) + Bd.dot(u[:,0].value)
        print ("time tkaen for 10 step: ", time.time()-t00)
            
    def generate_command(self):  
        """
        publishes the control command obtained from MPC solver
        """      
        if self.feasible == 0.0:
            print ("MPC CONTROLLER NOT FEASIBLE")
            self.uPred[:,0] = 0.0
            self.uPred[:,1] = -4.5

            self.xPred[:] = self.x0_local
            self.uPred.astype('float64')
            self.xPred.astype('float64')
            
        self.uminus1 = self.uPred[0,:] 

        ## MPC Command ##
        cmd_steer = self.uPred[0,0]
        cmd_acc   = self.uPred[0,1]
        cmd = [cmd_steer, cmd_acc]
        path_control = np.hstack([self.xPred,np.zeros([self.N+1, self.nx])])
        return cmd, path_control


