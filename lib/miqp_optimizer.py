import numpy as np
import gurobipy as gp
from gurobipy import GRB

class miqpOptimizer:
    def __init__(self, Qcal, qvec, A, lb, ub, n, m, method = "gurobi", warm_start = False, verbose = False):
        self.method = method
        self.warm_start = warm_start
        self.verbose = verbose

        if self.method not in ["cvxpy", "gurobi"]:
            raise ValueError("method must be 'cvxpy' or 'gurobi'.")

        self.Q = Qcal
        self.q = qvec
        self.A = A
        self.lb = lb
        self.ub = ub
        self.n, self.m = n, m
        self.Nxu = 2 * n + m
        self.nz = self.Q.shape[0]
        self.n_bin = self.nz - self.Nxu if self.nz > self.Nxu else 0
        self.z_prev = np.zeros(self.nz) 
        self.initialize_solver()
    
    def initialize_solver(self):
        if self.method == "cvxpy":
            import cvxpy as cp

        elif self.method == "gurobi":
            # initialize model
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', self.verbose)
                env.setParam('DualReductions', 0)
                env.start()
                self.model = gp.Model("qp", env=env)
            
            # --- add decision variables ---
            vtypes = [GRB.CONTINUOUS] * self.Nxu + [GRB.BINARY] * self.n_bin
            self.z = self.model.addMVar(shape=self.nz, vtype=vtypes,  lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z")
            
            # --- quadratic objective 0.5 z'Qz + q'z ---
            self.Q_obj = self.z.T@self.Q@self.z
            self.q_obj = self.q.T@self.z
            self.model_objective = self.Q_obj + self.q_obj
            self.model.setObjective(self.model_objective, GRB.MINIMIZE)

            # --- constraints: lb <= A z <= ub ---
            self.model_constr_lb = self.model.addConstr(self.A @ self.z >= self.lb, name="ineq_lower")
            self.model_constr_ub = self.model.addConstr(self.A @ self.z <= self.ub, name="ineq_upper")

    def update(self, q, lb, ub):
        if self.method == "cvxpy":
            pass
        elif self.method == "gurobi":
            self.q = q
            self.q_obj = self.q.T@self.z
            self.model_objective = self.Q_obj + self.q_obj
            self.model.setObjective(self.model_objective, GRB.MINIMIZE)
            self.model_constr_lb.RHS = self.lb
            self.model_constr_ub.RHS = self.ub
            if self.warm_start:
                self.z.start = self.z_prev

    def solve(self):
        if self.method == "cvxpy":
            pass
        elif self.method == "gurobi":    
            self.model.optimize()

            # --- extract solution ---
            if self.model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                z_val = self.z.X
                res = {
                    "status": 1 if self.model.Status == GRB.OPTIMAL else 0,
                    "obj": self.model.ObjVal,
                    "z": z_val,
                    "x0": z_val[:self.n],
                    "x1": z_val[self.n:2*self.n],
                    "u0": z_val[2*self.n:2*self.n + self.m],
                    "p0": z_val[2*self.n + self.m:2*self.n + self.m + self.n_bin]
                }
            else:
                res = {"status": 0, "obj": None, "z": None, "x0": None, "x1": None, "u0": None, "p0": None}
            return res
