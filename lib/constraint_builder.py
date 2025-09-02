# constraints_helper.py
import numpy as np
import polytope
from typing import List, Tuple
from helper import verts_to_hrep, inflate_obstacle_poly, deflate_workspace_poly

Array = np.ndarray
HRep = Tuple[Array, Array]  # (F, f) meaning F x <= f

class ConstraintHelper:
    """
    Build state and input constraints for state layout [x, y, vx, vy].

    - Safe state sets are AND'ed (stacked).
    - Unsafe convex obstacles are decomposed into facet-wise 'outside' disjuncts.
      NonConvexDecomposition(eps).build(...) returns Fx_list, fx_list.
    - Acceleration constraints (Gu, gu) are built directly in u-space from vertices.
    """

    def __init__(self, eps: float = 0.0, agent_radius = 0.0):
        self.eps = float(eps)
        self._pos_dim = 2
        self._vel_dim = 2
        self._state_dim = self._pos_dim + self._vel_dim
        self._agent_radius = agent_radius

    # ---- individual builders ----
    def workspace_constraints(self, workspace_vertices: Array) -> HRep:
        A, b = verts_to_hrep(workspace_vertices)          # A[x,y] <= b
        A, b = deflate_workspace_poly(A, b, self._agent_radius) # to not let robot go outside this region
        F = np.hstack([A, np.zeros((A.shape[0], self._vel_dim))])  # lift to [x,y,vx,vy]
        return F, b

    def velocity_constraints(self, vel_vertices: Array) -> HRep:
        A, b = verts_to_hrep(vel_vertices)                # A[vx,vy] <= b
        F = np.hstack([np.zeros((A.shape[0], self._pos_dim)), A])  # lift to [x,y,vx,vy]
        return F, b

    def obstacle_constraints(self, obstacle_vertices_list: List[Array]) -> List[HRep]:
        out = []
        for V in obstacle_vertices_list:
            A, b = verts_to_hrep(V)                       # A[x,y] <= b  (inside unsafe)
            A, b = inflate_obstacle_poly(A, b, self._agent_radius) # to not let robot inside obstacles
            F = np.hstack([A, np.zeros((A.shape[0], self._vel_dim))])  # lift
            out.append((F, b))
        return out

    def acceleration_constraints(self, accel_vertices: Array) -> HRep:
        # u-space constraints (no lifting), Gu*u <= gu with u=[ax, ay]
        return verts_to_hrep(accel_vertices)

    # ---- main helper ----
    def construct_state_constraints(
        self,
        workspace_vertices: Array,
        obstacle_vertices_list: List[Array],
        vel_vertices: Array,
        accel_vertices: Array,
    ):
        """
        Returns
        -------
        Fx, fx : Convex constraint in state
        Fx_hs_list, fx_hs_list : lists for disjunctive state constraints (half-spaces) (per obstacle, per facet)
        Gu, gu           : input-space constraints
        obs_Fx, obs_fx   : unsafe state: obstacle constraints (could be used for visualization)
        """
        F_w, f_w = self.workspace_constraints(workspace_vertices)
        F_v, f_v = self.velocity_constraints(vel_vertices)
        safe_state_sets = [(F_w, f_w), (F_v, f_v)]
        
        # construct list of polyhedron from vertices
        unsafe_state_sets = self.obstacle_constraints(obstacle_vertices_list)

        # decompose into halfsapces
        decomp = NonConvexDecomposition(eps=self.eps)
        Fx, fx, Fx_hs_list, fx_hs_list, obs_Fx, obs_fx = decomp.build(safe_state_sets, unsafe_state_sets)

        Gu, gu = self.acceleration_constraints(accel_vertices)
        return Fx, fx, Fx_hs_list, fx_hs_list, Gu, gu, obs_Fx, obs_fx


class NonConvexDecomposition:
    """
    Build disjunctive convex sets for 'safe ∩ (outside each unsafe polytope)'.

    Inputs:
      - Safe sets: list of H-reps (F, f). These are AND'ed (stacked).
      - Unsafe sets: list of H-reps (A, b) describing INSIDE of convex obstacles (unsafe).
                     For each obstacle k with rows i, we create facet-wise outside sets:
                         A_i x >= b_i (+ eps)
                     which in <= form becomes:
                         (-A_i) x <= -(b_i + eps)

    Output:
      - F_comb: List[List[(F_i, f_i)]]
        For each obstacle k: a list of facet-wise convex sets (F_i, f_i).
        In a MICP, you typically introduce binaries per obstacle to enforce
        'OR over i' while AND-ing safe constraints and all obstacles' disjunctions.
    """

    def __init__(self, eps: float = 0.0):
        """
        Parameters
        ----------
        eps : float
            Safety margin added to b when building 'outside' constraints A_i x >= b_i + eps.
        """
        self.eps = float(eps)


    def stack_safe(self, safe_sets):
        """
        Stack all safe constraints into a single H-rep (F, f).
        """
        if safe_sets is None or len(safe_sets) == 0:
            return None
        else:
            F_list, f_list = [], []
            for (F, f) in safe_sets:
                F_list.append(np.asarray(F, dtype=float))
                f_list.append(np.asarray(f, dtype=float).reshape(-1))
            F_safe = np.vstack(F_list)
            f_safe = np.hstack(f_list)
            return F_safe, f_safe

    def build(self, safe_sets = None, unsafe_sets = None):
        """
        Parameters
        ----------
        safe_sets  : List[(F, f)]
            List of safe convex sets in H-rep (F x <= f). These are AND'ed (stacked).
        unsafe_sets: List[(A, b)]
            List of unsafe convex sets (inside regions) in H-rep (A x <= b).

        Returns
        -------
        F_comb : List[List[(F_i, f_i)]]
            For each obstacle k, a list of facet-wise convex sets (F_i, f_i) where
            the first row encodes the 'outside' of facet i:
                (A_k[i]) x <= (b_k[i] + eps)
            followed by all stacked safe constraints.
        """
        # Stack all safe constraints once Fx * x <= fx
        if safe_sets is not None:
            safe_stacked = self.stack_safe(safe_sets)
            Fx, fx = safe_stacked
        else: 
            Fx, fx = None, None

        obs_Fx = []
        obs_fx = []
        ## Decompose nonconvex state safe sets into halfsapces
        if safe_stacked is not None and (unsafe_sets is not None or unsafe_sets == []):
            Fx_hs_list = []
            fx_hs_list = []
            for idx_obs, (A, b) in enumerate(unsafe_sets):
                A = np.asarray(A, dtype=float)
                b = np.asarray(b, dtype=float).reshape(-1)
                obs_Fx.append(A)
                obs_fx.append(b)
                m, d = A.shape
                Fx_hs_list_ = []
                fx_hs_list_ = []
                for i in range(m):
                    # Build 'outside' half-space for facet i in <= form
                    H_A = -A[i : i + 1, :]                   # Half- space shape (1, d)
                    h_b = -np.array([(b[i] + self.eps)])     # Half- space shape (1,)

                    if safe_stacked is not None:
                        F_safe, f_safe = safe_stacked
                        if F_safe.shape[1] != d:
                            raise ValueError(
                                f"Dim mismatch: safe dim {F_safe.shape[1]} vs obstacle dim {d}"
                            )
                        F_i = np.vstack([H_A, F_safe])
                        f_i = np.hstack([h_b, f_safe])
                    else:
                        F_i, f_i = H_A, h_b

                    Fx_hs_list_.append(F_i)
                    fx_hs_list_.append(f_i)
                Fx_hs_list.append(Fx_hs_list_)
                fx_hs_list.append(fx_hs_list_)
        else:
            Fx_hs_list = None
            fx_hs_list = None

        return Fx, fx, Fx_hs_list, fx_hs_list, obs_Fx, obs_fx

if __name__ == "__main__":
    import sys, os
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root) +'/')
    from config import config

    cfg = config()
    helper = ConstraintHelper(eps=0.0, agent_radius=cfg.agent_size)

    Workspace = cfg.Workspace
    Obs = cfg.Obs
    vel_vertices   = cfg.vel_vertices
    accel_vertices = cfg.accel_vertices

    Fx, fx, Fx_hs_list, fx_hs_list, Gu, gu, obs_Fx, obs_fx = helper.construct_state_constraints(
        workspace_vertices=Workspace,
        obstacle_vertices_list=Obs,
        vel_vertices=vel_vertices,
        accel_vertices=accel_vertices,
    )
    print("Fx:", Fx)
    print("fx:", fx)
    print("Fx_hs_list:", Fx_hs_list)
    print("fx_hs_list:", fx_hs_list)
    print("Gu:", Gu)
    print("gu:", gu)